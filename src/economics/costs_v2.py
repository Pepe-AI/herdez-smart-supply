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


def simulate_day_with_priority(
    df_day: pd.DataFrame,
    y_true: np.ndarray,
    scores: np.ndarray,
    config: CapacityConfig,
    y_proba: np.ndarray,
) -> dict:
    """Simula un día con priorización uniforme por scores.

    Aplica dos filtros económicos antes de priorizar:
    1. Umbral económico: p > config.economic_threshold (0.0187)
    2. Beneficio neto positivo: costo_esperado > costo_transferencia

    La probabilidad p viene del modelo ML (y_proba). El filtro
    económico requiere p para todas las estrategias; la
    priorización usa scores (que varían por estrategia).
    Esto garantiza que ninguna estrategia tome decisiones
    económicamente irracionales.

    Args:
        df_day: Datos del día (SKU x CEDI).
        y_true: Labels reales (1=quiebre).
        scores: Score de prioridad por fila (mayor = más prioritario).
        config: Configuración de capacidad.
        y_proba: Probabilidades del modelo para filtros económicos.

    Returns:
        Dict con n_transfers, cost_transfer, cost_fn, cost_total,
        transferred, origin_map.
    """
    order = np.argsort(-scores)

    capacity_used: dict[str, int] = {}
    transferred: set[tuple[str, str]] = set()
    origin_map: dict[tuple[str, str], str] = {}
    cost_transfer = 0.0

    for idx in order:
        idx_int = int(idx)
        row = df_day.iloc[idx_int]
        sku = str(row["sku_id"])
        cedi = str(row["cedi"])

        if (sku, cedi) in transferred:
            continue

        ventas_lag = float(row.get("ventas_rolling_7d_lag1", 0))
        ventas_proj = ventas_lag * config.dias_expuestos
        stock_lag = float(row.get("stock_lag_1", 0))
        deficit = max(0, ventas_proj - stock_lag)

        p = float(y_proba[idx_int])
        if p <= config.economic_threshold:
            continue
        cq = float(row["costo_quiebre_stock_diario"])
        costo_esp = p * cq * config.dias_expuestos
        ctu = float(row["costo_transferencia_unidad"])
        costo_tr_est = ctu * max(deficit, 50)
        if costo_esp - costo_tr_est <= 0:
            continue

        result = select_origin(
            sku, cedi, df_day, capacity_used, config
        )
        if result is None:
            continue

        cedi_origen, origin_units = result
        units = min(origin_units, max(int(deficit), 50))

        cost_transfer += (
            float(row["costo_transferencia_unidad"]) * units
        )
        transferred.add((sku, cedi))
        origin_map[(sku, cedi)] = cedi_origen
        capacity_used[cedi_origen] = (
            capacity_used.get(cedi_origen, 0) + 1
        )

    cost_fn = 0.0
    for i, (_, row) in enumerate(df_day.iterrows()):
        if (
            y_true[i] == 1
            and (row["sku_id"], row["cedi"]) not in transferred
        ):
            cost_fn += (
                float(row["costo_quiebre_stock_diario"])
                * config.dias_expuestos
            )

    return {
        "n_transfers": len(transferred),
        "cost_transfer": cost_transfer,
        "cost_fn": cost_fn,
        "cost_total": cost_transfer + cost_fn,
        "transferred": transferred,
        "origin_map": origin_map,
    }


STRATEGY_NAMES = [
    "inaction",
    "fifo",
    "by_stock_lag1",
    "heuristic_deficit",
    "model_prioritized",
    "by_costo_quiebre",
    "by_tasa_base",
]


def run_capacity_comparison(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int = 5,
) -> pd.DataFrame:
    """Compara 7 estrategias bajo restricción N=3 con TimeSeriesSplit.

    Todas las estrategias aplican los mismos filtros económicos
    antes de priorizar (umbral p > 0.0187 y beneficio_neto > 0).
    La probabilidad p viene del modelo ML para todas; la única
    diferencia entre estrategias es el criterio de ordenamiento
    dentro del conjunto ya filtrado. Esto garantiza comparación
    justa: ninguna estrategia toma decisiones irracionales.

    Correcciones aplicadas:
    - Bug 1: select_origin() uniforme para todas
    - Bug 2: by_stock_lag1 usa stock_lag_1 (no stock_actual)
    - Bug 3: filtros económicos uniformes (no solo para modelo)

    Estrategias:
    1. inaction: no transfiere nada
    2. fifo: orden de llegada (filas del DataFrame)
    3. by_stock_lag1: menor stock de ayer primero
    4. heuristic_deficit: mayor déficit proyectado primero
    5. model_prioritized: mayor beneficio_neto
    6. by_costo_quiebre: mayor costo de quiebre diario primero
    7. by_tasa_base: mayor tasa histórica de quiebre primero

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

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(df_train[feature_cols], y.iloc[tr_idx])
        y_proba = model.predict_proba(df_val[feature_cols])[:, 1]

        tasa_base_map = (
            df_train.groupby(["sku_id", "cedi"])["quiebre_proyectado"].mean()
        )

        dates = df_val["fecha"].unique()

        fold_results: dict[str, float] = {"fold": float(fold)}
        for name in STRATEGY_NAMES:
            fold_results[name] = 0.0

        for date in dates:
            mask = df_val["fecha"] == date
            df_day = df_val[mask].reset_index(drop=True)
            y_day = y_val[mask.values]
            proba_day = y_proba[mask.values]
            n = len(df_day)

            # 1. Inacción: solo costos FN
            for i in range(n):
                if y_day[i] == 1:
                    fold_results["inaction"] += (
                        float(df_day.iloc[i]["costo_quiebre_stock_diario"])
                        * config.dias_expuestos
                    )

            # 2. FIFO: orden de llegada
            fifo_scores = np.arange(n, 0, -1, dtype=float)
            fold_results["fifo"] += simulate_day_with_priority(
                df_day, y_day, fifo_scores, config,
                y_proba=proba_day,
            )["cost_total"]

            # 3. by_stock_lag1: menor stock de ayer (Bug 2)
            stock_scores = (
                -df_day["stock_lag_1"].values.astype(float)
            )
            fold_results["by_stock_lag1"] += (
                simulate_day_with_priority(
                    df_day, y_day, stock_scores, config,
                    y_proba=proba_day,
                )["cost_total"]
            )

            # 4. heuristic_deficit: mayor déficit proyectado
            deficit_scores = (
                df_day["ventas_rolling_7d_lag1"].values
                * config.dias_expuestos
                - df_day["stock_lag_1"].values
            ).astype(float)
            fold_results["heuristic_deficit"] += (
                simulate_day_with_priority(
                    df_day, y_day, deficit_scores, config,
                    y_proba=proba_day,
                )["cost_total"]
            )

            # 5. model_prioritized: beneficio_neto
            cq_v = (
                df_day["costo_quiebre_stock_diario"]
                .values.astype(float)
            )
            costo_esp = proba_day * cq_v * config.dias_expuestos
            vrl = (
                df_day["ventas_rolling_7d_lag1"]
                .values.astype(float)
            )
            v_proj = vrl * config.dias_expuestos
            s_lag = df_day["stock_lag_1"].values.astype(float)
            deficit_est = np.maximum(0, v_proj - s_lag)
            ctu = (
                df_day["costo_transferencia_unidad"]
                .values.astype(float)
            )
            costo_tr = ctu * np.maximum(deficit_est, 50)
            model_scores = costo_esp - costo_tr
            fold_results["model_prioritized"] += (
                simulate_day_with_priority(
                    df_day, y_day, model_scores, config,
                    y_proba=proba_day,
                )["cost_total"]
            )

            # 6. by_costo_quiebre: mayor costo quiebre diario
            cq_scores = (
                df_day["costo_quiebre_stock_diario"]
                .values.astype(float)
            )
            fold_results["by_costo_quiebre"] += (
                simulate_day_with_priority(
                    df_day, y_day, cq_scores, config,
                    y_proba=proba_day,
                )["cost_total"]
            )

            # 7. by_tasa_base: mayor tasa histórica quiebre
            tb_scores = np.array(
                [
                    tasa_base_map.get(
                        (row["sku_id"], row["cedi"]), 0.0
                    )
                    for _, row in df_day.iterrows()
                ]
            )
            fold_results["by_tasa_base"] += (
                simulate_day_with_priority(
                    df_day, y_day, tb_scores, config,
                    y_proba=proba_day,
                )["cost_total"]
            )

        results.append(fold_results)
        logger.info(
            f"Fold {fold}: "
            + " | ".join(
                f"{k}=${fold_results[k]:,.0f}" for k in STRATEGY_NAMES
            )
        )

    results_df = pd.DataFrame(results)

    logger.info("--- Resumen promedio por fold ---")
    for name in STRATEGY_NAMES:
        logger.info(f"  {name}: ${results_df[name].mean():,.0f}")

    return results_df
