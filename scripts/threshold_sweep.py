"""Threshold sweep: find optimal economic threshold for the ML model."""

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
df["deficit_estimado"] = (
    df["ventas_rolling_7d_lag1"] * 5 - df["stock_lag_1"]
).clip(lower=0)

df_clean = df.dropna(
    subset=["ventas_rolling_7d", "ventas_rolling_7d_lag1", "stock_lag_7", "coverage_ratio_lag"]
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
    tp_mask = (y_true == 1) & (y_pred == 1)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    units_moved = np.maximum(deficit_est, 50.0)
    tp_cost = float(np.sum(costo_t_unidad[tp_mask] * units_moved[tp_mask]))
    fp_cost = float(np.sum(costo_t_unidad[fp_mask] * units_moved[fp_mask]))
    fn_cost = float(np.sum(costo_q[fn_mask] * DIAS))
    cost_no_model = float(np.sum(costo_q[y_true == 1] * DIAS))
    return {
        "cost_total": tp_cost + fp_cost + fn_cost,
        "cost_no_model": cost_no_model,
        "n_transfers": int(np.sum(y_pred == 1)),
        "tp": int(np.sum(tp_mask)), "fp": int(np.sum(fp_mask)),
        "fn": int(np.sum(fn_mask)),
    }


# Theoretical optimum
print("=" * 70)
print("UMBRAL ECONOMICO TEORICO")
print("=" * 70)
print("costo_FP_promedio ~= $997  (transfer innecesaria)")
print("costo_FN_promedio ~= $53,333  (quiebre no detectado)")
print("Umbral optimo = costo_FP / (costo_FP + costo_FN)")
t_opt = 997 / (997 + 53333)
print(f"  = 997 / (997 + 53333) = {t_opt:.4f}")
print()

# Sweep
thresholds = [0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
results = {}

for t in thresholds:
    fold_costs = []
    for tr_idx, va_idx in tscv.split(df_eval):
        df_train = df_eval.iloc[tr_idx]
        df_val = df_eval.iloc[va_idx].reset_index(drop=True)
        y_val = y.iloc[va_idx].values
        costo_q = df_val["Costo_Quiebre_Stock_Diario"].values.astype(float)
        costo_t = df_val["Costo_Transferencia_Unidad"].values.astype(float)
        deficit = df_val["deficit_estimado"].values.astype(float)

        m = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            is_unbalance=True, verbosity=-1, random_state=42,
        )
        m.fit(df_train[feats], y.iloc[tr_idx])
        proba = m.predict_proba(df_val[feats])[:, 1]
        y_pred = (proba >= t).astype(int)
        fold_costs.append(compute_costs(y_val, y_pred, costo_q, costo_t, deficit))
    results[t] = fold_costs

# Also compute all_positive and inaction for reference
ref = {"all_positive": [], "inaction": []}
for tr_idx, va_idx in tscv.split(df_eval):
    df_val = df_eval.iloc[va_idx].reset_index(drop=True)
    y_val = y.iloc[va_idx].values
    costo_q = df_val["Costo_Quiebre_Stock_Diario"].values.astype(float)
    costo_t = df_val["Costo_Transferencia_Unidad"].values.astype(float)
    deficit = df_val["deficit_estimado"].values.astype(float)
    ref["all_positive"].append(
        compute_costs(y_val, np.ones(len(y_val), dtype=int), costo_q, costo_t, deficit))
    ref["inaction"].append(
        compute_costs(y_val, np.zeros(len(y_val), dtype=int), costo_q, costo_t, deficit))

ap_cost = np.mean([f["cost_total"] for f in ref["all_positive"]])
inaction_cost = np.mean([f["cost_total"] for f in ref["inaction"]])

print("=" * 95)
print("SWEEP DE UMBRALES — MODELO LAGGED (TimeSeriesSplit 5 folds)")
print("=" * 95)
print(f"{'Umbral':>8s} {'Costo/fold':>15s} {'vs inaccion':>15s} {'vs all_pos':>15s} {'Transfers':>10s} {'FN':>5s}")
print("-" * 75)

print(f"{'inaccion':>8s} ${inaction_cost:>12,.0f}   ${0:>12,.0f}   ${ap_cost-inaction_cost:>+12,.0f}   {'0':>8s}  {'-':>4s}")
print(f"{'all_pos':>8s} ${ap_cost:>12,.0f}   ${inaction_cost-ap_cost:>12,.0f}   ${0:>12,.0f}   {'176':>8s}  {'0':>4s}")
print("-" * 75)

sweep_costs = []
sweep_thresholds = []
for t in thresholds:
    folds_data = results[t]
    mean_cost = np.mean([f["cost_total"] for f in folds_data])
    std_cost = np.std([f["cost_total"] for f in folds_data])
    mean_transfers = np.mean([f["n_transfers"] for f in folds_data])
    mean_fn = np.mean([f["fn"] for f in folds_data])
    sav_inaction = inaction_cost - mean_cost
    sav_ap = ap_cost - mean_cost

    sweep_costs.append(mean_cost)
    sweep_thresholds.append(t)

    print(f"{t:>8.2f} ${mean_cost:>12,.0f}   ${sav_inaction:>12,.0f}   ${sav_ap:>+12,.0f}   {mean_transfers:>8.0f}  {mean_fn:>4.0f}")

# Find empirical optimum
best_idx = np.argmin(sweep_costs)
best_t = sweep_thresholds[best_idx]
best_cost = sweep_costs[best_idx]

print()
print(f"Umbral optimo empirico: {best_t} (costo ${best_cost:,.0f})")
print(f"Umbral optimo teorico:  {t_opt:.4f}")
print(f"Diferencia: {abs(best_t - t_opt)/t_opt*100:.0f}%")
print(f"Modelo con umbral optimo vs all_positive: ${ap_cost - best_cost:+,.0f}")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(sweep_thresholds, [c/1e6 for c in sweep_costs], "b-o", linewidth=2, label="Modelo LAGGED")
ax.axhline(y=ap_cost/1e6, color="green", linestyle="--", linewidth=2, label=f"All positive (${ap_cost/1e6:.2f}M)")
ax.axhline(y=inaction_cost/1e6, color="red", linestyle="--", linewidth=2, label=f"Inaccion (${inaction_cost/1e6:.2f}M)")
ax.axvline(x=t_opt, color="orange", linestyle=":", linewidth=1.5, label=f"Umbral teorico ({t_opt:.3f})")
ax.axvline(x=best_t, color="purple", linestyle=":", linewidth=1.5, label=f"Umbral empirico ({best_t})")
ax.set_xlabel("Umbral de decision")
ax.set_ylabel("Costo total por fold (M MXN)")
ax.set_title("Costo vs Umbral de Decision del Modelo LAGGED")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("reports/figures/threshold_sweep.png", dpi=150)
print("\nGrafico guardado en reports/figures/threshold_sweep.png")
