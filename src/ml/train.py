"""Entrenamiento del modelo LightGBM para predicción de quiebres.

Ejecutar como:
    python -m src.ml.train
"""

import logging
from typing import Any

import duckdb
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score

from src.data.config import get_settings
from src.data.features import build_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Parámetros del modelo (compartidos entre train y evaluate)
LGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "average_precision",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "is_unbalance": True,
    "random_state": 42,
    "verbosity": -1,
}

# Columnas que NO son features (se excluyen del entrenamiento)
EXCLUDE_COLS: list[str] = [
    "fecha",
    "sku_id",
    "cedi",
    "clima",  # Usamos clima_encoded en su lugar
    "quiebre_proyectado",  # Target
    "costo_quiebre_stock_diario",  # Info de costo, no feature predictiva
    "costo_transferencia_unidad",  # Info de costo, no feature predictiva
]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Retorna las columnas que se usarán como features."""
    return [col for col in df.columns if col not in EXCLUDE_COLS]


def train_model(df: pd.DataFrame) -> lgb.LGBMClassifier:
    """Entrena LightGBM sobre el DataFrame con features.

    Usa parámetros conservadores para un dataset pequeño (~1060 filas):
    - num_leaves=31: complejidad moderada
    - learning_rate=0.05: convergencia lenta pero estable
    - n_estimators=200: suficientes iteraciones con lr bajo
    - is_unbalance=True: compensa el desbalance 67/33
    """
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["quiebre_proyectado"]

    model = lgb.LGBMClassifier(**LGBM_PARAMS)

    model.fit(X, y)

    # Log métricas en train (referencia, no evaluación real)
    y_proba = model.predict_proba(X)[:, 1]
    train_auc_pr = average_precision_score(y, y_proba)
    logger.info("AUC-PR en train: %.4f", train_auc_pr)

    # Log feature importance (top 5)
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(
        ascending=False
    )
    logger.info("Top-5 features por importancia:")
    for feat, imp in importances.head(5).items():
        logger.info("  %s: %d", feat, imp)

    return model


def main() -> None:
    """Pipeline de entrenamiento completo."""
    settings = get_settings()

    # Cargar datos crudos de DuckDB
    logger.info("Cargando datos desde %s", settings.duckdb_path)
    with duckdb.connect(str(settings.duckdb_path), read_only=True) as con:
        df_raw = con.execute("SELECT * FROM inventory_history").fetchdf()

    # Generar features on-the-fly
    df = build_features(df_raw)

    # Entrenar modelo
    logger.info(
        "Entrenando LightGBM (%d filas, %d features)",
        len(df),
        len(get_feature_columns(df)),
    )
    model = train_model(df)

    # Persistir modelo
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(settings.model_path))
    logger.info("Modelo guardado en %s", settings.model_path)


if __name__ == "__main__":
    main()
