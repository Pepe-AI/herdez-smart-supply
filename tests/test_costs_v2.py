"""Tests para src/economics/costs_v2.py — simulación con restricción de capacidad."""

import numpy as np
import pandas as pd

from src.economics.costs_v2 import (
    CapacityConfig,
    compute_origin_safety,
    generate_alerts,
    prioritize_and_allocate,
    select_origin,
)

CONFIG = CapacityConfig(
    max_transfers_per_cedi=3,
    safety_factor=0.5,
    dias_expuestos=5,
    economic_threshold=0.0187,
)


def _make_day_df(n_skus: int = 2, n_cedis: int = 4) -> pd.DataFrame:
    """DataFrame sintético de un día: n_skus × n_cedis filas."""
    rows = []
    skus = [f"SKU_{i}" for i in range(n_skus)]
    cedis = [f"CEDI_{i}" for i in range(n_cedis)]
    for sku_idx, sku in enumerate(skus):
        for cedi_idx, cedi in enumerate(cedis):
            rows.append(
                {
                    "sku_id": sku,
                    "cedi": cedi,
                    "stock_actual": 500 + cedi_idx * 300,  # 500, 800, 1100, 1400
                    "stock_lag_1": 500 + cedi_idx * 300,
                    "ventas_rolling_7d_lag1": 100.0,
                    "lead_time_dias": 5,
                    "costo_quiebre_stock_diario": 15000 if sku_idx == 0 else 8000,
                    "costo_transferencia_unidad": 10.0,
                    "quiebre_proyectado": 0,
                }
            )
    return pd.DataFrame(rows)


class TestGenerateAlerts:
    def test_filters_below_threshold(self):
        df = _make_day_df()
        proba = np.array([0.01] * len(df))  # todo debajo del umbral
        alerts = generate_alerts(df, proba, CONFIG)
        assert len(alerts) == 0

    def test_includes_above_threshold(self):
        df = _make_day_df()
        proba = np.array([0.5] + [0.01] * (len(df) - 1))
        alerts = generate_alerts(df, proba, CONFIG)
        assert len(alerts) == 1
        assert alerts[0]["p_quiebre"] == 0.5

    def test_computes_expected_cost(self):
        df = _make_day_df()
        proba = np.array([0.8] + [0.01] * (len(df) - 1))
        alerts = generate_alerts(df, proba, CONFIG)
        # costo_esperado = 0.8 * 15000 * 5 = 60000
        assert alerts[0]["costo_esperado_no_actuar"] == 0.8 * 15000 * 5


class TestSelectOrigin:
    def test_picks_highest_surplus(self):
        df = _make_day_df(n_skus=1, n_cedis=4)
        # CEDI_3 tiene stock=1400, mayor surplus
        result = select_origin("SKU_0", "CEDI_0", df, {}, CONFIG)
        assert result is not None
        cedi_origen, units = result
        assert cedi_origen == "CEDI_3"  # mayor stock = mayor surplus

    def test_excludes_destination(self):
        df = _make_day_df(n_skus=1, n_cedis=2)
        # Solo 2 CEDIs: CEDI_0 (destino) y CEDI_1 (candidato)
        result = select_origin("SKU_0", "CEDI_0", df, {}, CONFIG)
        assert result is not None
        assert result[0] == "CEDI_1"

    def test_none_when_no_safe_origin(self):
        df = _make_day_df(n_skus=1, n_cedis=2)
        # Stock bajo → todos inseguros tras transferir
        df["stock_actual"] = 100
        df["stock_lag_1"] = 100
        df["ventas_rolling_7d_lag1"] = 200.0  # safety = 200*5*0.5=500 > 100
        result = select_origin("SKU_0", "CEDI_0", df, {}, CONFIG)
        assert result is None

    def test_respects_capacity(self):
        df = _make_day_df(n_skus=1, n_cedis=2)
        # CEDI_1 ya agotó capacidad
        capacity = {"CEDI_1": 3}
        result = select_origin("SKU_0", "CEDI_0", df, capacity, CONFIG)
        assert result is None  # único candidato agotado


class TestPrioritizeAndAllocate:
    def test_sorts_by_benefit(self):
        df = _make_day_df(n_skus=2, n_cedis=4)
        alerts = [
            {
                "sku_id": "SKU_0",
                "cedi": "CEDI_0",
                "p_quiebre": 0.9,
                "costo_esperado_no_actuar": 67500,
                "costo_transferir": 500,
                "beneficio_neto": 67000,
                "deficit_estimado": 50,
                "costo_quiebre_diario": 15000,
                "costo_transferencia_unidad": 10.0,
            },
            {
                "sku_id": "SKU_1",
                "cedi": "CEDI_0",
                "p_quiebre": 0.3,
                "costo_esperado_no_actuar": 12000,
                "costo_transferir": 500,
                "beneficio_neto": 11500,
                "deficit_estimado": 50,
                "costo_quiebre_diario": 8000,
                "costo_transferencia_unidad": 10.0,
            },
        ]
        approved, deferred = prioritize_and_allocate(alerts, df, CONFIG)
        assert len(approved) >= 1
        # Primera transferencia aprobada debe ser la de mayor beneficio
        assert approved[0].sku_id == "SKU_0"

    def test_respects_n3_capacity(self):
        df = _make_day_df(n_skus=1, n_cedis=4)
        # 4 alertas para 4 CEDIs destino, un solo SKU
        # Todas necesitan origen de otros CEDIs
        alerts = []
        for i in range(4):
            alerts.append(
                {
                    "sku_id": "SKU_0",
                    "cedi": f"CEDI_{i}",
                    "p_quiebre": 0.8,
                    "costo_esperado_no_actuar": 60000,
                    "costo_transferir": 500,
                    "beneficio_neto": 59500,
                    "deficit_estimado": 50,
                    "costo_quiebre_diario": 15000,
                    "costo_transferencia_unidad": 10.0,
                }
            )
        approved, deferred = prioritize_and_allocate(alerts, df, CONFIG)
        # Verificar que ningún CEDI origen tiene >3 transferencias
        origin_counts: dict[str, int] = {}
        for t in approved:
            origin_counts[t.cedi_origen] = origin_counts.get(t.cedi_origen, 0) + 1
        for count in origin_counts.values():
            assert count <= CONFIG.max_transfers_per_cedi

    def test_defers_negative_benefit(self):
        df = _make_day_df()
        alerts = [
            {
                "sku_id": "SKU_0",
                "cedi": "CEDI_0",
                "p_quiebre": 0.02,
                "costo_esperado_no_actuar": 1500,
                "costo_transferir": 5000,
                "beneficio_neto": -3500,
                "deficit_estimado": 500,
                "costo_quiebre_diario": 15000,
                "costo_transferencia_unidad": 10.0,
            }
        ]
        approved, deferred = prioritize_and_allocate(alerts, df, CONFIG)
        assert len(approved) == 0
        assert len(deferred) == 1


class TestOriginSafety:
    def test_safe_when_surplus_large(self):
        result = compute_origin_safety(
            stock_origen=1000,
            ventas_rolling_origen=100,
            lead_time=5,
            transfer_units=200,
            config=CONFIG,
        )
        # safety_threshold = 100 * 5 * 0.5 = 250
        # stock_after = 1000 - 200 = 800 >= 250 → safe
        assert result["is_safe"] is True
        assert result["stock_after"] == 800
        assert result["safety_threshold"] == 250

    def test_unsafe_when_surplus_small(self):
        result = compute_origin_safety(
            stock_origen=300,
            ventas_rolling_origen=100,
            lead_time=5,
            transfer_units=200,
            config=CONFIG,
        )
        # stock_after = 300 - 200 = 100 < 250 → unsafe
        assert result["is_safe"] is False
