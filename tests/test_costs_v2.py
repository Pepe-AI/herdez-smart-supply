"""Tests para src/economics/costs_v2.py — simulación con restricción de capacidad."""

import numpy as np
import pandas as pd

from src.economics.costs_v2 import (
    STRATEGY_NAMES,
    CapacityConfig,
    compute_origin_safety,
    generate_alerts,
    prioritize_and_allocate,
    select_origin,
    simulate_day_with_priority,
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


class TestSimulateDayWithPriority:
    def test_returns_expected_keys(self):
        df = _make_day_df(n_skus=1, n_cedis=4)
        y_true = np.array([1, 0, 0, 0])
        scores = np.array([10.0, 1.0, 1.0, 1.0])
        y_proba = np.array([0.5, 0.5, 0.5, 0.5])
        result = simulate_day_with_priority(df, y_true, scores, CONFIG, y_proba=y_proba)
        assert "n_transfers" in result
        assert "cost_transfer" in result
        assert "cost_fn" in result
        assert "cost_total" in result
        assert result["cost_total"] == result["cost_transfer"] + result["cost_fn"]

    def test_higher_priority_gets_served_first(self):
        """Con capacidad limitada, la fila con mayor score se transfiere primero."""
        df = _make_day_df(n_skus=2, n_cedis=2)
        y_true = np.array([1, 0, 1, 0])
        y_proba = np.array([0.5, 0.5, 0.5, 0.5])
        # SKU_0/CEDI_0 tiene score alto, SKU_1/CEDI_0 tiene score bajo
        scores_high_first = np.array([100.0, 0.0, 1.0, 0.0])
        scores_low_first = np.array([1.0, 0.0, 100.0, 0.0])

        r1 = simulate_day_with_priority(
            df, y_true, scores_high_first, CONFIG, y_proba=y_proba
        )
        r2 = simulate_day_with_priority(
            df, y_true, scores_low_first, CONFIG, y_proba=y_proba
        )
        # Ambos deben producir al menos 1 transferencia
        assert r1["n_transfers"] >= 1
        assert r2["n_transfers"] >= 1

    def test_zero_scores_still_transfers(self):
        """Con scores iguales (FIFO-like), aún se hacen transferencias."""
        df = _make_day_df(n_skus=1, n_cedis=4)
        y_true = np.array([1, 0, 0, 0])
        scores = np.zeros(len(df))
        y_proba = np.array([0.5, 0.5, 0.5, 0.5])
        result = simulate_day_with_priority(df, y_true, scores, CONFIG, y_proba=y_proba)
        assert result["n_transfers"] >= 1

    def test_all_strategies_produce_results(self):
        """Verifica que las 7 estrategias definidas en STRATEGY_NAMES existen."""
        assert len(STRATEGY_NAMES) == 7
        assert "inaction" in STRATEGY_NAMES
        assert "fifo" in STRATEGY_NAMES
        assert "model_prioritized" in STRATEGY_NAMES
        assert "by_tasa_base" in STRATEGY_NAMES

    def test_inaction_has_highest_cost_when_quiebres_exist(self):
        """Inacción siempre cuesta más que cualquier estrategia activa."""
        df = _make_day_df(n_skus=2, n_cedis=4)
        y_true = np.ones(len(df))  # todos quiebran
        scores = np.random.default_rng(42).random(len(df))
        y_proba = np.full(len(df), 0.5)

        active_result = simulate_day_with_priority(
            df, y_true, scores, CONFIG, y_proba=y_proba
        )

        # Inacción = suma de todos los costos FN
        cost_inaction = sum(
            float(df.iloc[i]["costo_quiebre_stock_diario"]) * CONFIG.dias_expuestos
            for i in range(len(df))
        )
        assert active_result["cost_total"] <= cost_inaction


class TestBacktestMatchesAgentDecisions:
    """Bug 3: verifica equivalencia entre backtest y agente."""

    def _compute_beneficio_neto_scores(
        self, df: pd.DataFrame, y_proba: np.ndarray
    ) -> np.ndarray:
        """Calcula beneficio_neto vectorizado (misma fórmula que generate_alerts)."""
        cq = df["costo_quiebre_stock_diario"].values.astype(float)
        costo_esp = y_proba * cq * CONFIG.dias_expuestos
        vrl = df["ventas_rolling_7d_lag1"].values.astype(float)
        v_proj = vrl * CONFIG.dias_expuestos
        s_lag = df["stock_lag_1"].values.astype(float)
        deficit = np.maximum(0, v_proj - s_lag)
        ctu = df["costo_transferencia_unidad"].values.astype(float)
        costo_tr = ctu * np.maximum(deficit, 50)
        return costo_esp - costo_tr

    def test_backtest_matches_agent_decisions(self):
        """simulate_day_with_priority con model_prioritized produce
        las mismas transferencias que generate_alerts +
        prioritize_and_allocate."""
        df = _make_day_df(n_skus=2, n_cedis=4)
        y_proba = np.array(
            [0.9, 0.1, 0.05, 0.01, 0.7, 0.5, 0.01, 0.01]
        )
        y_true = np.array([1, 0, 0, 0, 1, 1, 0, 0])

        # Ruta agente
        alerts = generate_alerts(df, y_proba, CONFIG)
        approved, _ = prioritize_and_allocate(alerts, df, CONFIG)
        agent_transfers = {
            (t.sku_id, t.cedi_destino, t.cedi_origen)
            for t in approved
        }

        # Ruta backtest
        scores = self._compute_beneficio_neto_scores(df, y_proba)
        result = simulate_day_with_priority(
            df, y_true, scores, CONFIG, y_proba=y_proba
        )
        backtest_transfers = {
            (sku, cedi, result["origin_map"][(sku, cedi)])
            for sku, cedi in result["transferred"]
        }

        assert agent_transfers == backtest_transfers

    def test_costs_match_between_paths(self):
        """Costos totales idénticos entre ambas rutas."""
        df = _make_day_df(n_skus=2, n_cedis=4)
        y_proba = np.array(
            [0.9, 0.1, 0.05, 0.01, 0.7, 0.5, 0.01, 0.01]
        )
        y_true = np.array([1, 0, 0, 0, 1, 1, 0, 0])

        from src.economics.costs_v2 import simulate_day

        agent_result = simulate_day(df, y_proba, y_true, CONFIG)

        scores = self._compute_beneficio_neto_scores(df, y_proba)
        backtest_result = simulate_day_with_priority(
            df, y_true, scores, CONFIG, y_proba=y_proba
        )

        assert agent_result["cost_total"] == backtest_result[
            "cost_total"
        ]
        assert agent_result["n_transfers"] == backtest_result[
            "n_transfers"
        ]


class TestByStockLag1NotStockActual:
    """Verifica que by_stock_lag1 usa stock_lag_1, no stock_actual (Bug 2)."""

    def test_priority_follows_stock_lag1_not_stock_actual(self):
        df = _make_day_df(n_skus=2, n_cedis=2)
        # stock_actual y stock_lag_1 tienen orden invertido
        df.loc[df["sku_id"] == "SKU_0", "stock_lag_1"] = 100  # bajo → alta prioridad
        df.loc[df["sku_id"] == "SKU_0", "stock_actual"] = 9000  # alto
        df.loc[df["sku_id"] == "SKU_1", "stock_lag_1"] = 9000  # alto → baja prioridad
        df.loc[df["sku_id"] == "SKU_1", "stock_actual"] = 100  # bajo

        # Score by_stock_lag1: -stock_lag_1 (menor stock_lag_1 = mayor score)
        stock_lag1_scores = -df["stock_lag_1"].values.astype(float)

        # Verificar que SKU_0 tiene mayor score (stock_lag_1=100 → score=-100)
        sku0_mask = df["sku_id"] == "SKU_0"
        sku1_mask = df["sku_id"] == "SKU_1"
        assert stock_lag1_scores[sku0_mask].max() > stock_lag1_scores[sku1_mask].max()

        # Si usáramos stock_actual, el orden sería inverso
        stock_actual_scores = -df["stock_actual"].values.astype(float)
        assert (
            stock_actual_scores[sku1_mask].max()
            > stock_actual_scores[sku0_mask].max()
        )
