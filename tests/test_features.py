"""Tests para src/data/features.py — feature engineering.

Todos los tests usan DataFrames sintéticos (nunca el Excel real)
para ser rápidos, deterministas e independientes de los datos.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.features import CLIMA_MAP, build_features


def _make_synthetic_df(n_days: int = 14, n_groups: int = 1) -> pd.DataFrame:
    """Crea un DataFrame sintético que imita la estructura de inventory_history.

    Genera `n_groups` combinaciones SKU-CEDI, cada una con `n_days` días.
    Valores de ventas y stock son deterministas para facilitar verificación manual.
    """
    rows = []
    skus = [f"SKU_{i}" for i in range(n_groups)]
    cedis = [f"CEDI_{i}" for i in range(n_groups)]

    for g in range(n_groups):
        for d in range(n_days):
            rows.append(
                {
                    "fecha": pd.Timestamp("2024-03-01") + pd.Timedelta(days=d),
                    "sku_id": skus[g],
                    "cedi": cedis[g],
                    "ventas_unidades": 100 + d * 10,  # 100, 110, 120, ...
                    "stock_actual": 500 - d * 20,  # 500, 480, 460, ...
                    "lead_time_dias": 5,
                    "promocion_activa": 0,
                    "precio_combustible_mxn": 23.5,
                    "clima": ["Despejado", "Lluvia", "Tormenta"][d % 3],
                    "costo_quiebre_stock_diario": 15000,
                    "costo_transferencia_unidad": 10.5,
                }
            )

    return pd.DataFrame(rows)


class TestRollingMean:
    """Verifica que la media móvil de 7 días se calcula correctamente."""

    def test_rolling_mean_values(self):
        """Primera fila válida es día 7 (lag_7 necesita 7 shifts)."""
        df = _make_synthetic_df(n_days=10)
        result = build_features(df)

        # Primera fila válida = día 7 (índice 7): rolling cubre días 1-7
        # ventas días 1-7 = [110,120,130,140,150,160,170], media = 140
        first_row = result.iloc[0]
        assert first_row["ventas_rolling_7d"] == pytest.approx(140.0)

    def test_rolling_drops_initial_nans(self):
        """build_features debe eliminar filas con NaN del rolling."""
        df = _make_synthetic_df(n_days=14)
        result = build_features(df)

        # 14 días - 7 primeros con NaN (min_periods=7, shift crea NaN en pos 0-6)
        # Pero el lag_7 necesita 7 shifts, así que se pierden 7 filas
        assert result["ventas_rolling_7d"].isna().sum() == 0


class TestTargetComputation:
    """Verifica el cálculo del target quiebre_proyectado."""

    def test_quiebre_when_stock_low(self):
        """stock=100, ventas_7d=25 → (100 - 25*5) = -25 < 0 → quiebre=1."""
        df = _make_synthetic_df(n_days=14)
        result = build_features(df)

        # Verificar que hay al menos un quiebre (stock baja linealmente)
        assert result["quiebre_proyectado"].sum() > 0

    def test_no_quiebre_when_stock_high(self):
        """Con stock muy alto, nunca debería haber quiebre."""
        df = _make_synthetic_df(n_days=14)
        # Sobreescribir stock a un valor altísimo
        df["stock_actual"] = 99999
        result = build_features(df)

        assert result["quiebre_proyectado"].sum() == 0

    def test_target_formula_exact(self):
        """Verifica la fórmula exacta: (stock - ventas_rolling_7d * 5) < 0."""
        df = _make_synthetic_df(n_days=14)
        result = build_features(df)

        # Recalcular manualmente para cada fila
        expected = (result["stock_actual"] - result["ventas_rolling_7d"] * 5) < 0
        np.testing.assert_array_equal(
            result["quiebre_proyectado"].values,
            expected.astype(int).values,
        )


class TestCoverageRatio:
    """Verifica el cálculo de ratios de cobertura."""

    def test_coverage_ratio_formula(self):
        """coverage_ratio = stock / (ventas_rolling_7d * lead_time_dias)."""
        df = _make_synthetic_df(n_days=14)
        result = build_features(df)

        expected = result["stock_actual"] / (
            result["ventas_rolling_7d"] * result["lead_time_dias"]
        )
        np.testing.assert_array_almost_equal(
            result["coverage_ratio"].values,
            expected.values,
        )

    def test_days_of_stock_formula(self):
        """days_of_stock = stock / ventas_rolling_7d."""
        df = _make_synthetic_df(n_days=14)
        result = build_features(df)

        expected = result["stock_actual"] / result["ventas_rolling_7d"]
        np.testing.assert_array_almost_equal(
            result["days_of_stock"].values,
            expected.values,
        )


class TestLabelEncoding:
    """Verifica la codificación de variables categóricas."""

    def test_clima_encoding_matches_map(self):
        """Cada valor de clima debe mapearse según CLIMA_MAP."""
        df = _make_synthetic_df(n_days=14)
        result = build_features(df)

        for _, row in result.iterrows():
            expected_code = CLIMA_MAP[row["clima"]]
            assert row["clima_encoded"] == expected_code

    def test_sku_cedi_encoded_are_integers(self):
        """sku_encoded y cedi_encoded deben ser enteros no negativos."""
        df = _make_synthetic_df(n_days=14, n_groups=3)
        result = build_features(df)

        assert result["sku_encoded"].dtype in (np.int8, np.int16, np.int32, np.int64)
        assert result["cedi_encoded"].dtype in (np.int8, np.int16, np.int32, np.int64)
        assert (result["sku_encoded"] >= 0).all()
        assert (result["cedi_encoded"] >= 0).all()


class TestNoPipelineNans:
    """Verifica que el pipeline completo no deja NaN residuales."""

    def test_no_nans_single_group(self):
        """Un solo grupo no debe tener NaN al final."""
        df = _make_synthetic_df(n_days=14, n_groups=1)
        result = build_features(df)
        assert result.isna().sum().sum() == 0

    def test_no_nans_multiple_groups(self):
        """Múltiples grupos tampoco deben tener NaN."""
        df = _make_synthetic_df(n_days=14, n_groups=4)
        result = build_features(df)
        assert result.isna().sum().sum() == 0
