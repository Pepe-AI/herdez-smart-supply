"""Precomputa datos de backtest para Streamlit.

Genera:
  data/backtest_results.parquet  — costo por fold por estrategia (7 estrategias)
  data/daily_predictions.parquet — predicciones + decisiones del agente por día

Ejecutar: python -m scripts.precompute_backtest
"""

import logging
from pathlib import Path

import duckdb
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.data.config import get_settings
from src.data.features import build_features
from src.economics.costs_v2 import (
    CapacityConfig,
    generate_alerts,
    prioritize_and_allocate,
    run_capacity_comparison,
)
from src.ml.train import LGBM_PARAMS, get_feature_columns

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data")


def _generate_daily_predictions(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int = 5,
) -> pd.DataFrame:
    """Genera predicciones out-of-fold + decisiones del agente por día."""
    config = CapacityConfig()
    df_eval = df.sort_values("fecha").reset_index(drop=True)
    y = df_eval["quiebre_proyectado"]
    tscv = TimeSeriesSplit(n_splits=n_splits)

    rows: list[dict] = []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(df_eval), 1):
        df_train = df_eval.iloc[tr_idx]
        df_val = df_eval.iloc[va_idx]
        y_val = y.iloc[va_idx].values

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(df_train[feature_cols], y.iloc[tr_idx])
        y_proba = model.predict_proba(df_val[feature_cols])[:, 1]

        dates = df_val["fecha"].unique()

        for date in dates:
            mask = df_val["fecha"] == date
            df_day = df_val[mask].reset_index(drop=True)
            y_day = y_val[mask.values]
            proba_day = y_proba[mask.values]

            alerts = generate_alerts(df_day, proba_day, config)
            approved, deferred = prioritize_and_allocate(
                alerts, df_day, config
            )

            approved_set = {
                (t.sku_id, t.cedi_destino) for t in approved
            }
            deferred_set = {
                (a["sku_id"], a["cedi"]) for a in deferred
            }

            for i, (_, row) in enumerate(df_day.iterrows()):
                sku = row["sku_id"]
                cedi = row["cedi"]
                key = (sku, cedi)

                if key in approved_set:
                    decision = "aprobada"
                    t = next(
                        t for t in approved
                        if t.sku_id == sku and t.cedi_destino == cedi
                    )
                    origen = t.cedi_origen
                    unidades = t.unidades
                    costo_transfer = t.costo_transferencia
                elif key in deferred_set:
                    decision = "diferida"
                    origen = None
                    unidades = 0
                    costo_transfer = 0.0
                else:
                    decision = "sin_alerta"
                    origen = None
                    unidades = 0
                    costo_transfer = 0.0

                rows.append(
                    {
                        "fold": fold,
                        "fecha": date,
                        "sku_id": sku,
                        "cedi": cedi,
                        "p_quiebre": float(proba_day[i]),
                        "y_true": int(y_day[i]),
                        "stock_lag_1": float(row.get("stock_lag_1", 0)),
                        "ventas_rolling_7d_lag1": float(
                            row.get("ventas_rolling_7d_lag1", 0)
                        ),
                        "costo_quiebre_diario": float(
                            row["costo_quiebre_stock_diario"]
                        ),
                        "decision": decision,
                        "cedi_origen": origen,
                        "unidades": unidades,
                        "costo_transferencia": costo_transfer,
                    }
                )

        logger.info(f"Fold {fold}: {len(dates)} días procesados")

    return pd.DataFrame(rows)


def main() -> None:
    settings = get_settings()

    logger.info("Cargando datos desde %s", settings.duckdb_path)
    with duckdb.connect(str(settings.duckdb_path), read_only=True) as con:
        df_raw = con.execute("SELECT * FROM inventory_history").fetchdf()

    df = build_features(df_raw)
    feature_cols = get_feature_columns(df)
    logger.info("%d filas, %d features", len(df), len(feature_cols))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("--- Backtest: 7 estrategias x 5 folds ---")
    backtest_df = run_capacity_comparison(df, feature_cols)
    backtest_path = OUTPUT_DIR / "backtest_results.parquet"
    backtest_df.to_parquet(backtest_path, index=False)
    logger.info("Guardado: %s (%d filas)", backtest_path, len(backtest_df))

    logger.info("--- Predicciones diarias ---")
    predictions_df = _generate_daily_predictions(df, feature_cols)
    predictions_path = OUTPUT_DIR / "daily_predictions.parquet"
    predictions_df.to_parquet(predictions_path, index=False)
    logger.info("Guardado: %s (%d filas)", predictions_path, len(predictions_df))


if __name__ == "__main__":
    main()
