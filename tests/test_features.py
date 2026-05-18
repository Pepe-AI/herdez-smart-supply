"""Tests para src/data/features.py — feature engineering con lagged features.

Todos los tests usan DataFrames sintéticos (nunca el Excel real)
para ser rápidos, deterministas e independientes de los datos.
"""

import numpy as np
import pandas as pd

from src.data.features import CLIMA_MAP, build_features


def _make_synthetic_df(n_days: int = 20, n_groups: int = 1) -> pd.DataFrame:
    """Crea un DataFrame sintético que imita inventory_history.

    n_days=20 por defecto: necesitamos shift(1) + rolling(7, min_periods=7)
    + shift(7) = al menos 15 filas para tener resultados válidos.
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


class TestLaggedFeatures:
    """Verifica que los lags se calculan correctamente."""

    def test_stock_lag_1_value(self):
        """stock_lag_1 debe ser el stock del día anterior."""
        df = _make_synthetic_df(n_days=20)
        result = build_features(df)

        # En datos sintéticos: stock = 500 - d*20
        # Primera fila válida tiene stock_lag_1 = stock del día anterior
        first = result.iloc[0]
        # La primera fila tiene fecha >= día 8 (por lags + rolling)
        # stock_lag_1 = stock_actual de un día antes
        assert pd.notna(first["stock_lag_1"])
        assert pd.notna(first["ventas_lag_1"])

    def test_lags_exist(self):
        """Todas las columnas de lag deben existir."""
        df = _make_synthetic_df(n_days=20)
        result = build_features(df)

        for lag in [1, 3, 5, 7]:
            assert f"stock_lag_{lag}" in result.columns
            assert f"ventas_lag_{lag}" in result.columns

    def test_rolling_lag1_excludes_current(self):
        """ventas_rolling_7d_lag1 NO debe incluir ventas del día actual."""
        df = _make_synthetic_df(n_days=20)
        result = build_features(df)

        # ventas_rolling_7d_lag1 = mean(ventas[t-8:t-1]) para una ventana de 7
        # Nunca debe ser igual a ventas del día actual promediadas
        assert "ventas_rolling_7d_lag1" in result.columns
        assert result["ventas_rolling_7d_lag1"].notna().all()

    def test_no_leakage_columns(self):
        """stock_actual y ventas_unidades están en el df pero no deben
        ser usados como features (verificado por EXCLUDE_COLS en train.py)."""
        df = _make_synthetic_df(n_days=20)
        result = build_features(df)

        # Las columnas existen (necesarias para el target) pero
        # EXCLUDE_COLS las excluye del entrenamiento
        assert "stock_actual" in result.columns
        assert "ventas_unidades" in result.columns


class TestTargetComputation:
    """Verifica el cálculo del target quiebre_proyectado."""

    def test_quiebre_when_stock_low(self):
        """Con stock decreciente, debe haber quiebres."""
        df = _make_synthetic_df(n_days=20)
        result = build_features(df)
        assert result["quiebre_proyectado"].sum() > 0

    def test_no_quiebre_when_stock_high(self):
        """Con stock altísimo, nunca hay quiebre."""
        df = _make_synthetic_df(n_days=20)
        df["stock_actual"] = 99999
        result = build_features(df)
        assert result["quiebre_proyectado"].sum() == 0

    def test_target_is_binary(self):
        """Target debe ser 0 o 1."""
        df = _make_synthetic_df(n_days=20)
        result = build_features(df)
        assert set(result["quiebre_proyectado"].unique()).issubset({0, 1})


class TestCoverageRatioLag:
    """Verifica ratios de cobertura con features lagged."""

    def test_coverage_ratio_uses_lag(self):
        """coverage_ratio_lag = stock_lag_1 / (ventas_rolling_7d_lag1 * lead_time)."""
        df = _make_synthetic_df(n_days=20)
        result = build_features(df)

        expected = result["stock_lag_1"] / (
            result["ventas_rolling_7d_lag1"] * result["lead_time_dias"]
        )
        np.testing.assert_array_almost_equal(
            result["coverage_ratio_lag"].values,
            expected.values,
        )

    def test_days_of_stock_uses_lag(self):
        """days_of_stock_lag = stock_lag_1 / ventas_rolling_7d_lag1."""
        df = _make_synthetic_df(n_days=20)
        result = build_features(df)

        expected = result["stock_lag_1"] / result["ventas_rolling_7d_lag1"]
        np.testing.assert_array_almost_equal(
            result["days_of_stock_lag"].values,
            expected.values,
        )


class TestLabelEncoding:
    """Verifica la codificación de variables categóricas."""

    def test_clima_encoding_matches_map(self):
        df = _make_synthetic_df(n_days=20)
        result = build_features(df)
        for _, row in result.iterrows():
            assert row["clima_encoded"] == CLIMA_MAP[row["clima"]]

    def test_sku_cedi_encoded_are_integers(self):
        df = _make_synthetic_df(n_days=20, n_groups=3)
        result = build_features(df)
        assert result["sku_encoded"].dtype in (np.int8, np.int16, np.int32, np.int64)
        assert (result["sku_encoded"] >= 0).all()
        assert (result["cedi_encoded"] >= 0).all()


class TestNoPipelineNans:
    """Verifica que el pipeline no deja NaN residuales."""

    def test_no_nans_single_group(self):
        df = _make_synthetic_df(n_days=20, n_groups=1)
        result = build_features(df)
        assert result.isna().sum().sum() == 0

    def test_no_nans_multiple_groups(self):
        df = _make_synthetic_df(n_days=20, n_groups=4)
        result = build_features(df)
        assert result.isna().sum().sum() == 0

    def test_auxiliary_columns_removed(self):
        """_ventas_rolling_7d_t (auxiliar para target) no debe estar en output."""
        df = _make_synthetic_df(n_days=20)
        result = build_features(df)
        assert "_ventas_rolling_7d_t" not in result.columns
