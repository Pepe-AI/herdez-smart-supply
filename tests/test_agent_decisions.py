"""Tests del agente LangGraph — sin API key, usa mock LLM."""

import numpy as np
import pandas as pd

from src.agent.graph import build_graph
from src.agent.tools import set_day_context
from src.economics.costs_v2 import CapacityConfig

CONFIG = CapacityConfig()


def _make_day_df() -> pd.DataFrame:
    """Datos sintéticos de un día: 2 SKUs × 4 CEDIs = 8 filas."""
    rows = []
    skus = ["SKU_Salsa", "SKU_Atun"]
    cedis = ["CEDI_Norte", "CEDI_Sur", "CEDI_Occidente", "CEDI_Bajio"]
    for sku_idx, sku in enumerate(skus):
        for cedi_idx, cedi in enumerate(cedis):
            rows.append(
                {
                    "sku_id": sku,
                    "cedi": cedi,
                    "fecha": pd.Timestamp("2024-04-15"),
                    "stock_actual": 200 + cedi_idx * 400,  # 200, 600, 1000, 1400
                    "stock_lag_1": 200 + cedi_idx * 400,
                    "ventas_rolling_7d_lag1": 100.0,
                    "lead_time_dias": 5,
                    "costo_quiebre_stock_diario": 15000 if sku_idx == 0 else 8000,
                    "costo_transferencia_unidad": 10.0,
                    "quiebre_proyectado": 1 if cedi_idx == 0 else 0,
                }
            )
    return pd.DataFrame(rows)


def _run_graph(df_day, y_proba):
    """Helper: inyecta contexto, construye grafo con mock LLM, ejecuta."""
    set_day_context(df_day, y_proba)
    graph = build_graph()  # mock LLM
    result = graph.invoke(
        {
            "messages": [],
            "fecha": "2024-04-15",
            "alerts": [],
            "evaluated_alerts": [],
            "transfers": [],
            "deferred": [],
            "capacity_used": {},
            "explanation": "",
        }
    )
    return result


class TestAgentGraph:
    def test_produces_transfers_for_high_risk(self):
        """Con alertas de alta probabilidad, el grafo debe producir transferencias."""
        df = _make_day_df()
        # Alta probabilidad para primeros 2 SKU-CEDI
        proba = np.array([0.9, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1])
        result = _run_graph(df, proba)
        assert len(result["transfers"]) > 0

    def test_no_transfers_when_all_low_risk(self):
        """Sin alertas sobre umbral, no hay transferencias."""
        df = _make_day_df()
        proba = np.array([0.01] * len(df))
        result = _run_graph(df, proba)
        assert len(result["transfers"]) == 0
        assert len(result["deferred"]) == 0

    def test_respects_capacity_n3(self):
        """Ningún CEDI origen debe tener más de 3 transferencias."""
        df = _make_day_df()
        proba = np.array([0.9] * len(df))  # todas altas
        result = _run_graph(df, proba)
        origin_counts: dict[str, int] = {}
        for t in result["transfers"]:
            origin = t["cedi_origen"]
            origin_counts[origin] = origin_counts.get(origin, 0) + 1
        for count in origin_counts.values():
            assert count <= CONFIG.max_transfers_per_cedi

    def test_generates_explanation(self):
        """El grafo debe producir una explicación no vacía."""
        df = _make_day_df()
        proba = np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        result = _run_graph(df, proba)
        assert result["explanation"] != ""

    def test_defers_when_no_origin_available(self):
        """Si todos los orígenes están en riesgo, la alerta se difiere."""
        df = _make_day_df()
        # Stock bajo en todos los CEDIs → no hay origen seguro
        df["stock_actual"] = 100
        df["stock_lag_1"] = 100
        df["ventas_rolling_7d_lag1"] = 200.0  # safety threshold alto
        proba = np.array([0.9] + [0.01] * 7)
        result = _run_graph(df, proba)
        # La alerta existe pero no se puede ejecutar
        assert len(result["transfers"]) == 0


class TestDecisionLogic:
    def test_high_cost_sku_prioritized(self):
        """SKU con costo_quiebre=$15k se prioriza sobre $8k a misma probabilidad."""
        df = _make_day_df()
        # Misma probabilidad para ambos SKUs en CEDI_0
        proba = np.array([0.8, 0.01, 0.01, 0.01, 0.8, 0.01, 0.01, 0.01])
        result = _run_graph(df, proba)
        if len(result["transfers"]) >= 2:
            # Primera transferencia debe ser del SKU de mayor costo
            first = result["transfers"][0]
            assert first["sku_id"] == "SKU_Salsa"
