"""Ingesta: Excel crudo → tabla DuckDB `inventory_history`.

Ejecutar como:
    python -m src.data.ingest
"""

import logging
from pathlib import Path

import duckdb
import pandas as pd

from src.data.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_excel(path: str | Path) -> pd.DataFrame:
    """Lee la hoja Historico_Inventarios del Excel y castea Fecha a date."""
    df = pd.read_excel(path, sheet_name="Historico_Inventarios")

    # Fecha viene como string 'YYYY-MM-DD'; convertir a date
    df["Fecha"] = pd.to_datetime(df["Fecha"]).dt.date

    return df


def validate(df: pd.DataFrame) -> None:
    """Validaciones básicas sobre los datos crudos."""
    if len(df) != 1200:
        raise ValueError(f"Se esperaban 1200 filas, hay {len(df)}")
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        raise ValueError(f"Hay {null_count} valores nulos inesperados")
    logger.info("Validación OK: %d filas, 0 nulls", len(df))


def write_to_duckdb(df: pd.DataFrame, db_path: str | Path) -> None:
    """Escribe el DataFrame a DuckDB como tabla `inventory_history`."""
    with duckdb.connect(str(db_path)) as con:
        # Registrar el DataFrame y crear tabla con tipos explícitos
        con.register("df_raw", df)
        con.execute("""
            CREATE OR REPLACE TABLE inventory_history AS
            SELECT
                CAST(Fecha AS DATE) AS fecha,
                CAST(SKU_ID AS VARCHAR) AS sku_id,
                CAST(CEDI AS VARCHAR) AS cedi,
                CAST(Ventas_Unidades AS INTEGER) AS ventas_unidades,
                CAST(Stock_Actual AS INTEGER) AS stock_actual,
                CAST(Lead_Time_Dias AS INTEGER) AS lead_time_dias,
                CAST(Promocion_Activa AS INTEGER) AS promocion_activa,
                CAST(Precio_Combustible_MXN AS DOUBLE) AS precio_combustible_mxn,
                CAST(Clima AS VARCHAR) AS clima,
                CAST(Costo_Quiebre_Stock_Diario AS INTEGER)
                    AS costo_quiebre_stock_diario,
                CAST(Costo_Transferencia_Unidad AS DOUBLE) AS costo_transferencia_unidad
            FROM df_raw
        """)

        # Verificar conteo
        count = con.execute("SELECT COUNT(*) FROM inventory_history").fetchone()[0]
        logger.info("Tabla inventory_history creada: %d filas en %s", count, db_path)


def main() -> None:
    """Pipeline de ingesta completo."""
    settings = get_settings()

    logger.info("Leyendo Excel: %s", settings.raw_excel_path)
    df = load_excel(settings.raw_excel_path)

    validate(df)

    logger.info("Escribiendo a DuckDB: %s", settings.duckdb_path)
    # Asegurar que el directorio padre existe
    settings.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    write_to_duckdb(df, settings.duckdb_path)


if __name__ == "__main__":
    main()
