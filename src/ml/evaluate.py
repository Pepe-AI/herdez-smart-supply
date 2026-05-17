"""Evaluación del modelo con TimeSeriesSplit y métricas monetizadas.

Ejecutar como:
    python -m src.ml.evaluate
"""

import logging

import duckdb
import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit

from src.data.config import get_settings
from src.data.features import build_features
from src.economics.costs import monetized_confusion_matrix
from src.ml.train import LGBM_PARAMS, get_feature_columns

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

N_SPLITS = 5


def evaluate_fold(
    model: lgb.LGBMClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    df_val: pd.DataFrame,
) -> dict[str, float]:
    """Evalúa un fold: calcula AUC-PR, precision, recall y costos.

    Args:
        model: Clasificador ya entrenado en el train del fold.
        X_val: Features del set de validación.
        y_val: Target del set de validación.
        df_val: DataFrame completo del fold (incluye columnas de costos).

    Returns:
        Dict con métricas del fold.
    """
    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    auc_pr = average_precision_score(y_val, y_proba)
    precision = precision_score(y_val, y_pred, zero_division=0.0)
    recall = recall_score(y_val, y_pred, zero_division=0.0)

    # Matriz monetizada usando los costos reales de cada observación
    y_true_arr: npt.NDArray[np.int_] = np.asarray(y_val, dtype=np.int_)
    costo_q: npt.NDArray[np.float64] = np.asarray(
        df_val["costo_quiebre_stock_diario"], dtype=np.float64
    )
    costo_t: npt.NDArray[np.float64] = np.asarray(
        df_val["costo_transferencia_unidad"], dtype=np.float64
    )
    matrix = monetized_confusion_matrix(
        y_true=y_true_arr,
        y_pred=y_pred,
        costo_quiebre_diario=costo_q,
        costo_transferencia=costo_t,
        dias_expuestos=5,
    )

    return {
        "auc_pr": auc_pr,
        "precision": precision,
        "recall": recall,
        "fn_cost_mxn": matrix.fn_cost,
        "fp_cost_mxn": matrix.fp_cost,
        "total_error_cost_mxn": matrix.total_error_cost,
        "savings_vs_no_model_mxn": matrix.savings_vs_no_model,
    }


def run_time_series_cv(df: pd.DataFrame) -> list[dict[str, float]]:
    """Ejecuta TimeSeriesSplit y retorna métricas por fold.

    Los datos se ordenan globalmente por fecha. TimeSeriesSplit garantiza
    que cada fold de validación es temporalmente posterior al de entrenamiento,
    evitando data leakage.
    """
    df = df.sort_values("fecha").reset_index(drop=True)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["quiebre_proyectado"]

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    results: list[dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        df_val = df.iloc[val_idx]

        # Entrenar modelo fresco para este fold
        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(X_train, y_train)

        fold_metrics = evaluate_fold(model, X_val, y_val, df_val)
        results.append(fold_metrics)

        savings = fold_metrics["savings_vs_no_model_mxn"]
        logger.info(
            f"Fold {fold_idx}: AUC-PR={fold_metrics['auc_pr']:.4f}  "
            f"Precision={fold_metrics['precision']:.4f}  "
            f"Recall={fold_metrics['recall']:.4f}  "
            f"Ahorro=${savings:,.0f} MXN"
        )

    return results


def print_summary(results: list[dict[str, float]]) -> None:
    """Imprime resumen de métricas: media ± std across folds."""
    metrics_df = pd.DataFrame(results)

    logger.info("")
    logger.info("=" * 60)
    logger.info("RESUMEN (%d folds con TimeSeriesSplit)", len(results))
    logger.info("=" * 60)

    for col in metrics_df.columns:
        mean = metrics_df[col].mean()
        std = metrics_df[col].std()
        if "mxn" in col:
            logger.info(f"  {col}: ${mean:,.0f} ± ${std:,.0f} MXN")
        else:
            logger.info("  %s: %.4f ± %.4f", col, mean, std)


def main() -> None:
    """Pipeline de evaluación completo."""
    settings = get_settings()

    # Cargar datos crudos y generar features
    logger.info("Cargando datos desde %s", settings.duckdb_path)
    with duckdb.connect(str(settings.duckdb_path), read_only=True) as con:
        df_raw = con.execute("SELECT * FROM inventory_history").fetchdf()

    df = build_features(df_raw)
    logger.info("Evaluando con TimeSeriesSplit (n_splits=%d)", N_SPLITS)

    # Ejecutar cross-validation temporal
    results = run_time_series_cv(df)

    # Resumen
    print_summary(results)


if __name__ == "__main__":
    main()
