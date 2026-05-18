"""Feature engineering: datos crudos → features para el modelo ML.

Estrategia anti-leakage: el target se calcula sobre el instante t,
pero todas las features usan solo información de t-1 o anterior (lags).
Esto evita que el modelo aprenda la fórmula del target en lugar de
patrones predictivos reales.

Todas las transformaciones se aplican por grupo (SKU-CEDI) para respetar
la independencia entre series.
"""

import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Mapeo ordinal para clima
CLIMA_MAP: dict[str, int] = {"Despejado": 0, "Lluvia": 1, "Tormenta": 2}


def _add_lagged_features(group: pd.DataFrame) -> pd.DataFrame:
    """Agrega features basadas en lags (t-k) a un grupo SKU-CEDI.

    Todas las features miran al pasado para evitar leakage con el target en t.
    """
    group = group.copy()

    # Lags de stock (t-1, t-3, t-5, t-7)
    for lag in [1, 3, 5, 7]:
        group[f"stock_lag_{lag}"] = group["stock_actual"].shift(lag)
        group[f"ventas_lag_{lag}"] = group["ventas_unidades"].shift(lag)

    # Rolling de ventas sobre t-1 a t-7 (excluye t)
    group["ventas_rolling_7d_lag1"] = (
        group["ventas_unidades"].shift(1).rolling(window=7, min_periods=7).mean()
    )
    group["ventas_std_7d_lag1"] = (
        group["ventas_unidades"].shift(1).rolling(window=7, min_periods=7).std()
    )

    # Rolling de stock sobre t-1 a t-7
    group["stock_rolling_7d_lag1"] = (
        group["stock_actual"].shift(1).rolling(window=7, min_periods=7).mean()
    )

    # Rolling de ventas en t (solo para calcular el target, NO es feature)
    group["_ventas_rolling_7d_t"] = (
        group["ventas_unidades"].rolling(window=7, min_periods=7).mean()
    )

    return group


def _add_coverage_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula ratios de cobertura con features lagged.

    Usa stock_lag_1 y ventas_rolling_7d_lag1 (ambos del pasado).
    """
    df = df.copy()

    safe_ventas = df["ventas_rolling_7d_lag1"].replace(0, float("nan"))
    df["coverage_ratio_lag"] = df["stock_lag_1"] / (safe_ventas * df["lead_time_dias"])
    df["days_of_stock_lag"] = df["stock_lag_1"] / safe_ventas

    return df


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica variables categóricas como enteros para LightGBM."""
    df = df.copy()

    df["clima_encoded"] = df["clima"].map(CLIMA_MAP)
    if df["clima_encoded"].isna().any():
        unknown = df.loc[df["clima_encoded"].isna(), "clima"].unique()
        raise ValueError(f"Valores de clima desconocidos: {unknown}")

    df["sku_encoded"] = df["sku_id"].astype("category").cat.codes
    df["cedi_encoded"] = df["cedi"].astype("category").cat.codes

    return df


def _add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula el target: quiebre proyectado.

    Fórmula: (stock_actual_t - ventas_rolling_7d_t * 5) < 0
    Usa _ventas_rolling_7d_t (calculada sobre ventas en t) solo para el target.
    """
    df = df.copy()
    df["quiebre_proyectado"] = (
        (df["stock_actual"] - df["_ventas_rolling_7d_t"] * 5) < 0
    ).astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo de feature engineering.

    Args:
        df: DataFrame crudo con columnas de inventory_history (snake_case).

    Returns:
        DataFrame con features lagged, encodings, y target.
        Columnas auxiliares (_ventas_rolling_7d_t) se eliminan.
    """
    df = df.sort_values(["sku_id", "cedi", "fecha"]).reset_index(drop=True)

    # Lags y rolling por grupo
    groups = []
    for _, group in df.groupby(["sku_id", "cedi"]):
        group = _add_lagged_features(group)
        groups.append(group)
    df = pd.concat(groups, ignore_index=True)

    # Ratios de cobertura (dependen de lags)
    df = _add_coverage_ratios(df)

    # Encoding de categóricas
    df = _encode_categoricals(df)

    # Target
    df = _add_target(df)

    # Eliminar columnas auxiliares (usadas solo para el target)
    df = df.drop(columns=["_ventas_rolling_7d_t"])

    # Eliminar filas con NaN (lags + rolling requieren historia)
    rows_before = len(df)
    df = df.dropna().reset_index(drop=True)
    rows_dropped = rows_before - len(df)
    logger.info(
        "Features construidas: %d filas (%d eliminadas por NaN de lags/rolling)",
        len(df),
        rows_dropped,
    )

    return df
