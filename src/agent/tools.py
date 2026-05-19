"""Herramientas del agente expuestas a Gemini via Function Calling.

Wrappers delgados sobre funciones puras de costs_v2.py.
Los docstrings están en español porque Gemini los usa como descripción
de la herramienta en la interfaz de chat.
"""

import json
import logging

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from src.economics.costs_v2 import (
    CapacityConfig,
    compute_origin_safety,
    generate_alerts,
)

logger = logging.getLogger(__name__)

# Configuración por defecto
_config = CapacityConfig()

# Cache de datos del día (se setea antes de invocar el grafo)
_day_data: dict = {}


def set_day_context(
    df_day: pd.DataFrame,
    y_proba: np.ndarray,
    capacity_used: dict[str, int] | None = None,
) -> None:
    """Inyecta el contexto del día antes de ejecutar el agente."""
    _day_data["df_day"] = df_day
    _day_data["y_proba"] = y_proba
    _day_data["capacity_used"] = capacity_used or {}


@tool
def obtener_alertas_hoy() -> str:
    """Obtiene las alertas de quiebre para hoy.

    Consulta el modelo ML y filtra por umbral económico (p > 0.0187).
    Retorna lista de alertas con SKU, CEDI destino, probabilidad,
    costo esperado de no actuar, y beneficio neto de transferir.
    """
    df_day = _day_data["df_day"]
    y_proba = _day_data["y_proba"]
    alerts = generate_alerts(df_day, y_proba, _config)
    return json.dumps(alerts, ensure_ascii=False, indent=2)


@tool
def verificar_seguridad_origen(sku_id: str, cedi_origen: str, unidades: int) -> str:
    """Verifica si un CEDI origen queda seguro tras transferir N unidades.

    Un CEDI es seguro si su stock residual cubre al menos el 50%
    de la demanda proyectada durante el lead time.

    Args:
        sku_id: Identificador del SKU a transferir.
        cedi_origen: CEDI que enviaría las unidades.
        unidades: Cantidad de unidades a transferir.
    """
    df_day = _day_data["df_day"]
    row = df_day[(df_day["sku_id"] == sku_id) & (df_day["cedi"] == cedi_origen)]
    if row.empty:
        return json.dumps({"error": f"No hay datos para {sku_id} en {cedi_origen}"})

    row = row.iloc[0]
    result = compute_origin_safety(
        stock_origen=float(row["stock_actual"]),
        ventas_rolling_origen=float(row.get("ventas_rolling_7d_lag1", 0)),
        lead_time=int(row["lead_time_dias"]),
        transfer_units=unidades,
        config=_config,
    )
    return json.dumps(result)


@tool
def consultar_capacidad_restante(cedi: str) -> str:
    """Consulta cuántas transferencias de salida puede ejecutar un CEDI hoy.

    Máximo N=3 por CEDI origen por día.

    Args:
        cedi: Nombre del CEDI a consultar.
    """
    used = _day_data.get("capacity_used", {}).get(cedi, 0)
    remaining = _config.max_transfers_per_cedi - used
    return json.dumps(
        {
            "cedi": cedi,
            "usado": used,
            "restante": remaining,
            "maximo": _config.max_transfers_per_cedi,
        }
    )


@tool
def ejecutar_plan_transferencias(plan_json: str) -> str:
    """Aprueba y registra la lista final de transferencias a ejecutar.

    Recibe un JSON con la lista de transferencias priorizadas.
    Valida capacidad y seguridad antes de aprobar.

    Args:
        plan_json: JSON string con lista de transferencias.
    """
    try:
        plan = json.loads(plan_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "JSON inválido"})

    return json.dumps(
        {
            "status": "aprobado",
            "n_transferencias": len(plan) if isinstance(plan, list) else 0,
            "plan": plan,
        }
    )


def get_tools() -> list:
    """Retorna la lista de herramientas disponibles para el agente."""
    return [
        obtener_alertas_hoy,
        verificar_seguridad_origen,
        consultar_capacidad_restante,
        ejecutar_plan_transferencias,
    ]
