"""Tests para src/economics/costs.py — monetización de la matriz de confusión."""

import numpy as np
import pytest

from src.economics.costs import monetized_confusion_matrix


@pytest.fixture
def sample_costs():
    """Costos de ejemplo: 4 observaciones con quiebre real en posiciones 0 y 1."""
    return {
        "y_true": np.array([1, 1, 0, 0]),
        "costo_quiebre_diario": np.array([15000.0, 8000.0, 15000.0, 8000.0]),
        "costo_transferencia": np.array([12.0, 10.0, 11.0, 9.0]),
        "dias_expuestos": 5,
    }


class TestModeloPerfecto:
    """Modelo que detecta todos los quiebres sin falsos positivos."""

    def test_no_fn_no_fp(self, sample_costs):
        """Sin FN ni FP: todo TP y TN."""
        y_pred = np.array([1, 1, 0, 0])  # Predicción perfecta
        result = monetized_confusion_matrix(
            y_true=sample_costs["y_true"],
            y_pred=y_pred,
            costo_quiebre_diario=sample_costs["costo_quiebre_diario"],
            costo_transferencia=sample_costs["costo_transferencia"],
            dias_expuestos=sample_costs["dias_expuestos"],
        )

        assert result.fn_count == 0
        assert result.fp_count == 0
        assert result.tp_count == 2
        assert result.tn_count == 2
        assert result.fn_cost == 0.0
        assert result.fp_cost == 0.0
        # TP cost = transferencias correctas (12 + 10)
        assert result.tp_cost == pytest.approx(22.0)

    def test_savings_positive(self, sample_costs):
        """Modelo perfecto debe tener ahorro positivo."""
        y_pred = np.array([1, 1, 0, 0])
        result = monetized_confusion_matrix(
            y_true=sample_costs["y_true"],
            y_pred=y_pred,
            costo_quiebre_diario=sample_costs["costo_quiebre_diario"],
            costo_transferencia=sample_costs["costo_transferencia"],
            dias_expuestos=sample_costs["dias_expuestos"],
        )

        # cost_no_model = (15000 + 8000) * 5 = 115000
        assert result.cost_no_model == pytest.approx(115_000.0)
        # savings = 115000 - (22 + 0 + 0) = 114978
        assert result.savings_vs_no_model == pytest.approx(114_978.0)
        assert result.savings_vs_no_model > 0


class TestModeloTerrible:
    """Modelo que no detecta ningún quiebre (predice todo 0)."""

    def test_all_fn(self, sample_costs):
        """Todos los quiebres se pierden como FN."""
        y_pred = np.array([0, 0, 0, 0])
        result = monetized_confusion_matrix(
            y_true=sample_costs["y_true"],
            y_pred=y_pred,
            costo_quiebre_diario=sample_costs["costo_quiebre_diario"],
            costo_transferencia=sample_costs["costo_transferencia"],
            dias_expuestos=sample_costs["dias_expuestos"],
        )

        assert result.fn_count == 2
        assert result.tp_count == 0
        assert result.fp_count == 0
        # FN cost = (15000 + 8000) * 5 = 115000
        assert result.fn_cost == pytest.approx(115_000.0)

    def test_zero_savings(self, sample_costs):
        """Sin modelo = modelo terrible: ahorro debe ser 0."""
        y_pred = np.array([0, 0, 0, 0])
        result = monetized_confusion_matrix(
            y_true=sample_costs["y_true"],
            y_pred=y_pred,
            costo_quiebre_diario=sample_costs["costo_quiebre_diario"],
            costo_transferencia=sample_costs["costo_transferencia"],
            dias_expuestos=sample_costs["dias_expuestos"],
        )

        # cost_no_model = fn_cost = 115000, tp+fp = 0 → ahorro = 0
        assert result.savings_vs_no_model == pytest.approx(0.0)


class TestModeloAleatorio:
    """Modelo que acierta algunos y falla otros."""

    def test_mixed_predictions(self, sample_costs):
        """Predice quiebre en posiciones 1 y 2 (1 TP, 1 FP, 1 FN, 1 TN)."""
        y_pred = np.array([0, 1, 1, 0])
        result = monetized_confusion_matrix(
            y_true=sample_costs["y_true"],
            y_pred=y_pred,
            costo_quiebre_diario=sample_costs["costo_quiebre_diario"],
            costo_transferencia=sample_costs["costo_transferencia"],
            dias_expuestos=sample_costs["dias_expuestos"],
        )

        assert result.tp_count == 1
        assert result.fp_count == 1
        assert result.fn_count == 1
        assert result.tn_count == 1

        # TP cost: transferencia[1] = 10
        assert result.tp_cost == pytest.approx(10.0)
        # FP cost: transferencia[2] = 11
        assert result.fp_cost == pytest.approx(11.0)
        # FN cost: quiebre[0] * 5 = 15000 * 5 = 75000
        assert result.fn_cost == pytest.approx(75_000.0)

    def test_savings_partial(self, sample_costs):
        """Ahorro parcial: evita un quiebre pero paga transferencia innecesaria."""
        y_pred = np.array([0, 1, 1, 0])
        result = monetized_confusion_matrix(
            y_true=sample_costs["y_true"],
            y_pred=y_pred,
            costo_quiebre_diario=sample_costs["costo_quiebre_diario"],
            costo_transferencia=sample_costs["costo_transferencia"],
            dias_expuestos=sample_costs["dias_expuestos"],
        )

        # cost_no_model = (15000 + 8000) * 5 = 115000
        # cost_with_model = 10 + 11 + 75000 = 75021
        # savings = 115000 - 75021 = 39979
        assert result.savings_vs_no_model == pytest.approx(39_979.0)
        assert result.total_error_cost == pytest.approx(75_011.0)


class TestArrayVacio:
    """Caso borde: sin observaciones."""

    def test_empty_arrays(self):
        """Arrays vacíos deben retornar ceros en todo."""
        result = monetized_confusion_matrix(
            y_true=np.array([], dtype=np.int_),
            y_pred=np.array([], dtype=np.int_),
            costo_quiebre_diario=np.array([], dtype=np.float64),
            costo_transferencia=np.array([], dtype=np.float64),
            dias_expuestos=5,
        )

        assert result.tp_count == 0
        assert result.fp_count == 0
        assert result.fn_count == 0
        assert result.tn_count == 0
        assert result.tp_cost == 0.0
        assert result.fp_cost == 0.0
        assert result.fn_cost == 0.0
        assert result.cost_no_model == 0.0
        assert result.savings_vs_no_model == 0.0
        assert result.total_error_cost == 0.0
