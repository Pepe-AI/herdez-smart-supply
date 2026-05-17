"""Configuración centralizada del proyecto vía variables de entorno / .env."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración del proyecto. Valores se leen de .env o env vars."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Rutas de datos
    duckdb_path: Path = Path("data/herdez.db")
    raw_excel_path: Path = Path("data/raw/herdez_inventario.xlsx.xlsx")

    # Ruta del modelo entrenado
    model_path: Path = Path("models/lgbm_quiebre.txt")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton de configuración (se lee una sola vez)."""
    return Settings()
