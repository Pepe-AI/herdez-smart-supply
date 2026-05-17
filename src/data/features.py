"""Feature engineering: datos crudos → features para el modelo ML.

Todas las transformaciones se aplican por grupo (SKU-CEDI) para respetar
la independencia entre series. Las rolling windows usan min_periods=7
para evitar estimaciones con pocos datos.
"""

import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Mapeo ordinal para clima (placeholder — sin señal real según EDA)
CLIMA_MAP: dict[str, int] = {"Despejado": 0, "Lluvia": 1, "Tormenta": 2}


def _add_rolling_features(group: pd.DataFrame) -> pd.DataFrame:
    """Agrega features de ventana móvil a un grupo SKU-CEDI.

    Se calcula sobre el grupo ya ordenado por fecha.
    """
    group = group.copy()

    # Rolling de ventas (ventana=7, min_periods=7 para no usar datos parciales)
    group["ventas_rolling_7d"] = (
        group["ventas_unidades"].rolling(window=7, min_periods=7).mean()
    )
    group["ventas_std_7d"] = (
        group["ventas_unidades"].rolling(window=7, min_periods=7).std()
    )

    # Rolling de stock
    group["stock_rolling_7d"] = (
        group["stock_actual"].rolling(window=7, min_periods=7).mean()
    )

    return group


def _add_lag_features(group: pd.DataFrame) -> pd.DataFrame:
    """Agrega features de rezago (lags) a un grupo SKU-CEDI."""
    group = group.copy()

    group["ventas_lag_1"] = group["ventas_unidades"].shift(1)
    group["ventas_lag_7"] = group["ventas_unidades"].shift(7)

    return group


def _add_coverage_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula ratios de cobertura de inventario.

    - coverage_ratio: ¿cuántos lead times cubre el stock actual?
    - days_of_stock: ¿para cuántos días alcanza el stock?
    """
    df = df.copy()

    # Evitar división por cero (ventas_rolling_7d podría ser 0 en teoría)
    safe_ventas = df["ventas_rolling_7d"].replace(0, float("nan"))

    df["coverage_ratio"] = df["stock_actual"] / (safe_ventas * df["lead_time_dias"])
    df["days_of_stock"] = df["stock_actual"] / safe_ventas

    return df


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica variables categóricas como enteros para LightGBM."""
    df = df.copy()

    # Clima: ordinal encoding (es un placeholder sin señal)
    df["clima_encoded"] = df["clima"].map(CLIMA_MAP)
    if df["clima_encoded"].isna().any():
        unknown = df.loc[df["clima_encoded"].isna(), "clima"].unique()
        raise ValueError(f"Valores de clima desconocidos: {unknown}")

    # SKU y CEDI: label encoding (LightGBM los usará como categorías)
    df["sku_encoded"] = df["sku_id"].astype("category").cat.codes
    df["cedi_encoded"] = df["cedi"].astype("category").cat.codes

    return df


def _add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula el target: quiebre proyectado a 5 días.

    Fórmula (de CLAUDE.md):
        quiebre_proyectado = (stock_actual - ventas_rolling_7d * 5) < 0

    Retorna 1 si hay riesgo de quiebre, 0 si no.
    """
    df = df.copy()
    df["quiebre_proyectado"] = (
        (df["stock_actual"] - df["ventas_rolling_7d"] * 5) < 0
    ).astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo de feature engineering.

    Args:
        df: DataFrame crudo con columnas de inventory_history (snake_case).

    Returns:
        DataFrame con features calculadas y sin filas NaN.
        Filas con NaN (primeros 6 días por grupo, por ventana=7) se eliminan.
    """
    # Asegurar orden temporal dentro de cada grupo
    df = df.sort_values(["sku_id", "cedi", "fecha"]).reset_index(drop=True)

    # Rolling y lags por grupo (requieren orden temporal estricto)
    # En pandas 3.x, apply() con group_keys=False puede eliminar columnas
    # de agrupamiento del resultado. Usamos include_groups=False y concat.
    groups = []
    for _, group in df.groupby(["sku_id", "cedi"]):
        group = _add_rolling_features(group)
        group = _add_lag_features(group)
        groups.append(group)
    df = pd.concat(groups, ignore_index=True)

    # Ratios (dependen de rolling, se aplican globalmente)
    df = _add_coverage_ratios(df)

    # Encoding de categóricas
    df = _encode_categoricals(df)

    # Target
    df = _add_target(df)

    # Eliminar filas con NaN (primeros días sin suficiente historia)
    rows_before = len(df)
    df = df.dropna().reset_index(drop=True)
    rows_dropped = rows_before - len(df)
    logger.info(
        "Features construidas: %d filas (%d eliminadas por NaN de rolling)",
        len(df),
        rows_dropped,
    )

    return df
