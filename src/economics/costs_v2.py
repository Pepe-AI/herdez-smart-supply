"""Simulación económica con restricción de capacidad logística.

Módulo puro: zero imports de LangGraph/LangChain.
Toda la lógica de decisión del agente vive aquí como funciones puras.
El agente LangGraph es un wrapper delgado sobre estas funciones.

Restricción clave: máximo N=3 transferencias/día por CEDI origen.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CapacityConfig:
    """Parámetros de la simulación con restricciones."""

    max_transfers_per_cedi: int = 3
    safety_factor: float = 0.5  # retener 50% del excedente proyectado
    dias_expuestos: int = 5
    economic_threshold: float = 0.0187  # p > costo_FP / (costo_FP + costo_FN)


@dataclass(frozen=True)
class TransferProposal:
    """Una transferencia aprobada por la lógica de priorización."""

    sku_id: str
    cedi_destino: str
    cedi_origen: str
    p_quiebre: float
    beneficio_neto: float
    unidades: int
    costo_transferencia: float


def generate_alerts(
    df_day: pd.DataFrame,
    y_proba: np.ndarray,
    config: CapacityConfig,
) -> list[dict]:
    """Genera alertas para un día filtrando por umbral económico.

    Args:
        df_day: Datos de un día (hasta 20 filas: 5 SKUs × 4 CEDIs).
        y_proba: Probabilidades del modelo (misma longitud que df_day).
        config: Configuración de capacidad.

    Returns:
        Lista de alertas con campos: sku_id, cedi, p_quiebre,
        costo_esperado_no_actuar, deficit_estimado.
    """
    alerts = []
    for i, (_, row) in enumerate(df_day.iterrows()):
        p = float(y_proba[i])
        if p <= config.economic_threshold:
            continue

        costo_quiebre = float(row["costo_quiebre_stock_diario"])
        costo_esperado = p * costo_quiebre * config.dias_expuestos

        # Déficit estimado con info lagged
        ventas_proj = (
            float(row.get("ventas_rolling_7d_lag1", 0)) * config.dias_expuestos
        )
        stock_lag = float(row.get("stock_lag_1", 0))
        deficit = max(0, ventas_proj - stock_lag)

        costo_transfer = float(row["costo_transferencia_unidad"]) * max(deficit, 50)
        beneficio = costo_esperado - costo_transfer

        alerts.append(
            {
                "sku_id": row["sku_id"],
                "cedi": row["cedi"],
                "p_quiebre": p,
                "costo_esperado_no_actuar": costo_esperado,
                "costo_transferir": costo_transfer,
                "beneficio_neto": beneficio,
                "deficit_estimado": deficit,
                "costo_quiebre_diario": costo_quiebre,
                "costo_transferencia_unidad": float(row["costo_transferencia_unidad"]),
            }
        )

    return alerts


def compute_origin_safety(
    stock_origen: float,
    ventas_rolling_origen: float,
    lead_time: int,
    transfer_units: int,
    config: CapacityConfig,
) -> dict:
    """Verifica si el CEDI origen queda seguro tras transferir.

    Seguro = stock residual >= ventas_rolling * lead_time * safety_factor.

    Returns:
        Dict con is_safe, stock_after, safety_threshold.
    """
    stock_after = stock_origen - transfer_units
    safety_threshold = ventas_rolling_origen * lead_time * config.safety_factor
    return {
        "is_safe": stock_after >= safety_threshold,
        "stock_after": stock_after,
        "safety_threshold": safety_threshold,
    }


def select_origin(
    sku_id: str,
    cedi_destino: str,
    df_day: pd.DataFrame,
    capacity_used: dict[str, int],
    config: CapacityConfig,
) -> tuple[str, int] | None:
    """Selecciona el mejor CEDI origen para una transferencia.

    Criterio: mayor excedente que no quede en riesgo y tenga capacidad.

    Returns:
        (cedi_origen, unidades_a_mover) o None si no hay origen viable.
    """
    # Candidatos: otros CEDIs con el mismo SKU
    candidates = df_day[(df_day["sku_id"] == sku_id) & (df_day["cedi"] != cedi_destino)]

    best_origin = None
    best_units = 0
    best_surplus = -1.0

    for _, row in candidates.iterrows():
        cedi = row["cedi"]

        # Verificar capacidad
        used = capacity_used.get(cedi, 0)
        if used >= config.max_transfers_per_cedi:
            continue

        stock = float(row["stock_actual"])
        ventas_r = float(row.get("ventas_rolling_7d_lag1", 0))
        lead_time = int(row["lead_time_dias"])

        # Calcular excedente disponible (50% del surplus)
        safety_stock = ventas_r * lead_time * config.safety_factor
        surplus = stock - safety_stock
        if surplus <= 0:
            continue

        transferable = int(surplus * 0.5)
        if transferable <= 0:
            continue

        # Verificar que queda seguro
        safety = compute_origin_safety(stock, ventas_r, lead_time, transferable, config)
        if not safety["is_safe"]:
            continue

        if surplus > best_surplus:
            best_surplus = surplus
            best_origin = cedi
            best_units = transferable

    if best_origin is None:
        return None
    return (best_origin, best_units)


def prioritize_and_allocate(
    alerts: list[dict],
    df_day: pd.DataFrame,
    config: CapacityConfig,
) -> tuple[list[TransferProposal], list[dict]]:
    """Asignación greedy con restricción de capacidad.

    Ordena alertas por beneficio_neto desc, asigna a orígenes
    respetando N=max_transfers_per_cedi.

    Returns:
        (transferencias_aprobadas, alertas_diferidas)
    """
    # Ordenar por beneficio neto descendente
    sorted_alerts = sorted(alerts, key=lambda a: a["beneficio_neto"], reverse=True)

    approved: list[TransferProposal] = []
    deferred: list[dict] = []
    capacity_used: dict[str, int] = {}

    for alert in sorted_alerts:
        if alert["beneficio_neto"] <= 0:
            deferred.append(alert)
            continue

        result = select_origin(
            alert["sku_id"], alert["cedi"], df_day, capacity_used, config
        )

        if result is None:
            deferred.append(alert)
            continue

        cedi_origen, units = result
        # Limitar unidades al déficit estimado
        units = min(units, max(int(alert["deficit_estimado"]), 50))
        costo = alert["costo_transferencia_unidad"] * units

        approved.append(
            TransferProposal(
                sku_id=alert["sku_id"],
                cedi_destino=alert["cedi"],
                cedi_origen=cedi_origen,
                p_quiebre=alert["p_quiebre"],
                beneficio_neto=alert["beneficio_neto"],
                unidades=units,
                costo_transferencia=costo,
            )
        )

        # Consumir capacidad del origen
        capacity_used[cedi_origen] = capacity_used.get(cedi_origen, 0) + 1

    return approved, deferred


def simulate_day(
    df_day: pd.DataFrame,
    y_proba: np.ndarray,
    y_true: np.ndarray,
    config: CapacityConfig,
) -> dict:
    """Simula un día completo: alertas → priorización → costos.

    Returns:
        Dict con transfers, deferred, cost_total, cost_fn, cost_fp_tp.
    """
    alerts = generate_alerts(df_day, y_proba, config)
    approved, deferred = prioritize_and_allocate(alerts, df_day, config)

    # Costo de transferencias aprobadas (TP + FP)
    transferred_cedis = {(t.sku_id, t.cedi_destino) for t in approved}
    cost_transfer = sum(t.costo_transferencia for t in approved)

    # Costo de quiebres no cubiertos (FN)
    cost_fn = 0.0
    for i, (_, row) in enumerate(df_day.iterrows()):
        if y_true[i] == 1 and (row["sku_id"], row["cedi"]) not in transferred_cedis:
            cost_fn += float(row["costo_quiebre_stock_diario"]) * config.dias_expuestos

    return {
        "n_transfers": len(approved),
        "n_deferred": len(deferred),
        "cost_transfer": cost_transfer,
        "cost_fn": cost_fn,
        "cost_total": cost_transfer + cost_fn,
    }


def _simulate_strategy_random_truncated(
    df_day: pd.DataFrame,
    y_true: np.ndarray,
    config: CapacityConfig,
) -> dict:
    """Baseline: transferir todo, truncar random a N por CEDI."""
    capacity_used: dict[str, int] = {}
    transferred: set[tuple[str, str]] = set()
    cost_transfer = 0.0

    # Orden arbitrario (filas como vienen)
    for _, row in df_day.iterrows():
        sku = row["sku_id"]
        cedi = row["cedi"]

        # Buscar cualquier origen con capacidad
        origins = df_day[(df_day["sku_id"] == sku) & (df_day["cedi"] != cedi)]
        for _, orig_row in origins.iterrows():
            orig_cedi = orig_row["cedi"]
            if capacity_used.get(orig_cedi, 0) >= config.max_transfers_per_cedi:
                continue
            # Transferir
            deficit = max(
                float(row.get("ventas_rolling_7d_lag1", 0)) * config.dias_expuestos
                - float(row.get("stock_lag_1", 0)),
                50,
            )
            cost_transfer += float(row["costo_transferencia_unidad"]) * deficit
            capacity_used[orig_cedi] = capacity_used.get(orig_cedi, 0) + 1
            transferred.add((sku, cedi))
            break

    # FN: quiebres no cubiertos
    cost_fn = 0.0
    for i, (_, row) in enumerate(df_day.iterrows()):
        if y_true[i] == 1 and (row["sku_id"], row["cedi"]) not in transferred:
            cost_fn += float(row["costo_quiebre_stock_diario"]) * config.dias_expuestos

    return {
        "n_transfers": len(transferred),
        "cost_transfer": cost_transfer,
        "cost_fn": cost_fn,
        "cost_total": cost_transfer + cost_fn,
    }


def run_capacity_comparison(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int = 5,
) -> pd.DataFrame:
    """Compara estrategias bajo restricción N=3 con TimeSeriesSplit.

    Estrategias:
    1. inaction: predice todo 0
    2. all_positive_random: transfiere todo, trunca random a N=3/CEDI
    3. model_prioritized: usa probabilidades para priorizar top-N

    Returns:
        DataFrame con costo por fold para cada estrategia.
    """
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit

    from src.ml.train import LGBM_PARAMS

    config = CapacityConfig()
    df_eval = df.sort_values("fecha").reset_index(drop=True)
    y = df_eval["quiebre_proyectado"]
    tscv = TimeSeriesSplit(n_splits=n_splits)

    results: list[dict] = []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(df_eval), 1):
        df_train = df_eval.iloc[tr_idx]
        df_val = df_eval.iloc[va_idx]
        y_val = y.iloc[va_idx].values

        # Entrenar modelo
        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(df_train[feature_cols], y.iloc[tr_idx])
        y_proba = model.predict_proba(df_val[feature_cols])[:, 1]

        # Agrupar por fecha
        dates = df_val["fecha"].unique()

        fold_results: dict[str, float] = {
            "fold": fold,
            "inaction": 0.0,
            "all_positive_random": 0.0,
            "model_prioritized": 0.0,
        }

        for date in dates:
            mask = df_val["fecha"] == date
            df_day = df_val[mask].reset_index(drop=True)
            y_day = y_val[mask.values]
            proba_day = y_proba[mask.values]

            # 1. Inacción
            cost_inaction = float(
                np.sum(
                    df_day.loc[y_day == 1, "costo_quiebre_stock_diario"]
                    * config.dias_expuestos
                )
            )
            fold_results["inaction"] += cost_inaction

            # 2. All positive random truncated
            day_random = _simulate_strategy_random_truncated(df_day, y_day, config)
            fold_results["all_positive_random"] += day_random["cost_total"]

            # 3. Model prioritized
            day_model = simulate_day(df_day, proba_day, y_day, config)
            fold_results["model_prioritized"] += day_model["cost_total"]

        results.append(fold_results)
        logger.info(
            f"Fold {fold}: inaction=${fold_results['inaction']:,.0f}  "
            f"random=${fold_results['all_positive_random']:,.0f}  "
            f"model=${fold_results['model_prioritized']:,.0f}"
        )

    results_df = pd.DataFrame(results)

    # Headline number
    mean_random = results_df["all_positive_random"].mean()
    mean_model = results_df["model_prioritized"].mean()
    if mean_random > 0:
        pct_improvement = (mean_random - mean_model) / mean_random * 100
        logger.info(
            f"HEADLINE: modelo reduce costo {pct_improvement:.0f}%% vs "
            f"random truncated (${mean_model:,.0f} vs ${mean_random:,.0f})"
        )

    return results_df
