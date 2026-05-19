"""Compare cost strategies with realistic transfer volumes."""

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_excel(
    "data/raw/herdez_inventario.xlsx.xlsx", sheet_name="Historico_Inventarios"
)
df["Fecha"] = pd.to_datetime(df["Fecha"])
df = df.sort_values(["SKU_ID", "CEDI", "Fecha"]).reset_index(drop=True)

df["ventas_rolling_7d"] = df.groupby(["SKU_ID", "CEDI"])["Ventas_Unidades"].transform(
    lambda x: x.rolling(7, min_periods=7).mean()
)
df["quiebre"] = ((df["Stock_Actual"] - df["ventas_rolling_7d"] * 5) < 0).astype(int)
for lag in [1, 3, 5, 7]:
    df[f"stock_lag_{lag}"] = df.groupby(["SKU_ID", "CEDI"])["Stock_Actual"].shift(lag)
    df[f"ventas_lag_{lag}"] = df.groupby(["SKU_ID", "CEDI"])[
        "Ventas_Unidades"
    ].shift(lag)
df["ventas_rolling_7d_lag1"] = df.groupby(["SKU_ID", "CEDI"])[
    "Ventas_Unidades"
].transform(lambda x: x.shift(1).rolling(7, min_periods=7).mean())
df["ventas_std_7d_lag1"] = df.groupby(["SKU_ID", "CEDI"])[
    "Ventas_Unidades"
].transform(lambda x: x.shift(1).rolling(7, min_periods=7).std())
df["stock_rolling_7d_lag1"] = df.groupby(["SKU_ID", "CEDI"])[
    "Stock_Actual"
].transform(lambda x: x.shift(1).rolling(7, min_periods=7).mean())
safe_v = df["ventas_rolling_7d_lag1"].replace(0, float("nan"))
df["coverage_ratio_lag"] = df["stock_lag_1"] / (safe_v * df["Lead_Time_Dias"])
df["days_of_stock_lag"] = df["stock_lag_1"] / safe_v
df["sku_enc"] = df["SKU_ID"].astype("category").cat.codes
df["cedi_enc"] = df["CEDI"].astype("category").cat.codes
df["clima_enc"] = df["Clima"].map({"Despejado": 0, "Lluvia": 1, "Tormenta": 2})

# Deficit estimado con info de t-1 (lo que el sistema sabria al decidir)
df["deficit_estimado"] = (
    df["ventas_rolling_7d_lag1"] * 5 - df["stock_lag_1"]
).clip(lower=0)

df_clean = df.dropna(
    subset=[
        "ventas_rolling_7d", "ventas_rolling_7d_lag1",
        "stock_lag_7", "coverage_ratio_lag",
    ]
).reset_index(drop=True)

feats = [
    "stock_lag_1", "stock_lag_3", "stock_lag_5", "stock_lag_7",
    "ventas_lag_1", "ventas_lag_3", "ventas_lag_5", "ventas_lag_7",
    "ventas_rolling_7d_lag1", "ventas_std_7d_lag1", "stock_rolling_7d_lag1",
    "coverage_ratio_lag", "days_of_stock_lag",
    "Lead_Time_Dias", "Promocion_Activa", "Precio_Combustible_MXN",
    "Costo_Transferencia_Unidad", "sku_enc", "cedi_enc", "clima_enc",
]

df_eval = df_clean.sort_values("Fecha").reset_index(drop=True)
y = df_eval["quiebre"]
tscv = TimeSeriesSplit(n_splits=5)
DIAS = 5


def compute_costs(y_true, y_pred, costo_q, costo_t_unidad, deficit_est):
    """Costos con unidades reales transferidas.

    - TP/FP: se transfieren deficit_est unidades a costo_t_unidad cada una
    - FN: quiebre no detectado, costo_q * DIAS
    - Si deficit_est es 0 pero y_pred=1, se transfiere un minimo de seguridad
    """
    tp_mask = (y_true == 1) & (y_pred == 1)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)

    # Costo de transferencia = unidades * costo por unidad
    # Minimo 50 unidades cuando se decide transferir (lote minimo logistico)
    units_moved = np.maximum(deficit_est, 50.0)

    tp_cost = float(np.sum(costo_t_unidad[tp_mask] * units_moved[tp_mask]))
    fp_cost = float(np.sum(costo_t_unidad[fp_mask] * units_moved[fp_mask]))
    fn_cost = float(np.sum(costo_q[fn_mask] * DIAS))

    cost_no_model = float(np.sum(costo_q[y_true == 1] * DIAS))
    cost_total = tp_cost + fp_cost + fn_cost

    return {
        "cost_total": cost_total,
        "tp_cost": tp_cost, "fp_cost": fp_cost, "fn_cost": fn_cost,
        "cost_no_model": cost_no_model,
        "savings_vs_inaction": cost_no_model - cost_total,
        "n_transfers": int(np.sum(y_pred == 1)),
    }


strategies = {
    "inaccion_total": [],
    "all_positive": [],
    "heuristic_lag1": [],
    "tasa_base_gate05": [],
    "lgbm_lagged": [],
}

for fold, (tr_idx, va_idx) in enumerate(tscv.split(df_eval), 1):
    df_train = df_eval.iloc[tr_idx]
    df_val = df_eval.iloc[va_idx].reset_index(drop=True)
    y_val = y.iloc[va_idx].values
    costo_q = df_val["Costo_Quiebre_Stock_Diario"].values.astype(float)
    costo_t = df_val["Costo_Transferencia_Unidad"].values.astype(float)
    deficit = df_val["deficit_estimado"].values.astype(float)

    # 0. Inaccion total
    strategies["inaccion_total"].append(
        compute_costs(y_val, np.zeros(len(y_val), dtype=int), costo_q, costo_t, deficit)
    )

    # 1. All positive
    strategies["all_positive"].append(
        compute_costs(y_val, np.ones(len(y_val), dtype=int), costo_q, costo_t, deficit)
    )

    # 2. Heuristica
    heur_pred = (df_val["stock_lag_1"] - df_val["ventas_lag_1"] * 5 < 0).astype(int).values
    strategies["heuristic_lag1"].append(
        compute_costs(y_val, heur_pred, costo_q, costo_t, deficit)
    )

    # 3. Tasa base con gate 0.5
    rate_map = df_train.groupby(["SKU_ID", "CEDI"])["quiebre"].mean().to_dict()
    fallback = y.iloc[tr_idx].mean()
    proba_base = (
        df_val.set_index(["SKU_ID", "CEDI"])
        .index.map(lambda x: rate_map.get(x, fallback))
        .values.astype(float)
    )
    base_pred = (proba_base >= 0.5).astype(int)
    strategies["tasa_base_gate05"].append(
        compute_costs(y_val, base_pred, costo_q, costo_t, deficit)
    )

    # 4. LightGBM
    m = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        is_unbalance=True, verbosity=-1, random_state=42,
    )
    m.fit(df_train[feats], y.iloc[tr_idx])
    lgbm_pred = (m.predict_proba(df_val[feats])[:, 1] >= 0.5).astype(int)
    strategies["lgbm_lagged"].append(
        compute_costs(y_val, lgbm_pred, costo_q, costo_t, deficit)
    )

# === TABLE ===
print("=" * 105)
print("COSTOS CON UNIDADES REALES (deficit_estimado, min 50 unidades)")
print("=" * 105)

summary = {}
for name, folds_data in strategies.items():
    costs = [f["cost_total"] for f in folds_data]
    sav = [f["savings_vs_inaction"] for f in folds_data]
    summary[name] = {
        "cost": np.mean(costs), "cost_std": np.std(costs),
        "sav_inaction": np.mean(sav), "sav_inaction_std": np.std(sav),
        "tp": np.mean([f["tp_cost"] for f in folds_data]),
        "fp": np.mean([f["fp_cost"] for f in folds_data]),
        "fn": np.mean([f["fn_cost"] for f in folds_data]),
        "transfers": np.mean([f["n_transfers"] for f in folds_data]),
    }

ap_cost = summary["all_positive"]["cost"]

print(f"{'Estrategia':25s} {'Costo total':>18s} {'Ahorro vs inaccion':>22s} {'vs all_pos':>15s} {'Transfers':>10s}")
print("-" * 105)
for name, s in summary.items():
    sav_ap = ap_cost - s["cost"]
    print(
        f"{name:25s} ${s['cost']:>12,.0f}+/-{s['cost_std']:>8,.0f}  "
        f"${s['sav_inaction']:>12,.0f}+/-{s['sav_inaction_std']:>8,.0f}  "
        f"${sav_ap:>12,.0f}  "
        f"{s['transfers']:>8.0f}"
    )

print()
print("Desglose (medias):")
print(f"{'Estrategia':25s} {'TP cost':>12s} {'FP cost':>12s} {'FN cost':>12s}")
print("-" * 65)
for name, s in summary.items():
    print(f"{name:25s} ${s['tp']:>10,.0f} ${s['fp']:>10,.0f} ${s['fn']:>10,.0f}")

print()
ap_sav = summary["all_positive"]["sav_inaction"]
lgbm_sav = summary["lgbm_lagged"]["sav_inaction"]
lgbm_vs_ap = ap_cost - summary["lgbm_lagged"]["cost"]
heur_vs_ap = ap_cost - summary["heuristic_lag1"]["cost"]

print(f"all_positive ahorra vs inaccion: ${ap_sav:,.0f}")
print(f"lgbm_lagged ahorra vs inaccion: ${lgbm_sav:,.0f}")
print(f"Modelo vs all_positive: ${lgbm_vs_ap:+,.0f}")
print(f"Heuristica vs all_positive: ${heur_vs_ap:+,.0f}")
print()
print(f"Costo FP promedio por transferencia (all_positive): ${summary['all_positive']['fp']/summary['all_positive']['transfers']:.0f}")
print(f"Costo FN promedio por quiebre no detectado: ${summary['lgbm_lagged']['fn']/(176*0.57):.0f}")
