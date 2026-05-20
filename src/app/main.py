"""Dashboard Herdez Smart-Supply.

Ejecutar: streamlit run src/app/main.py
"""

import logging
from pathlib import Path

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

_INITIAL_STATE = {
    "messages": [],
    "fecha": "",
    "alerts": [],
    "evaluated_alerts": [],
    "transfers": [],
    "deferred": [],
    "capacity_used": {},
    "explanation": "",
}

DATA_DIR = Path("data")
FIGURES_DIR = Path("reports/figures")


@st.cache_data
def load_backtest() -> pd.DataFrame | None:
    path = DATA_DIR / "backtest_results.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data
def load_predictions() -> pd.DataFrame | None:
    path = DATA_DIR / "daily_predictions.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


STRATEGY_LABELS = {
    "inaction": "Inaccion",
    "fifo": "FIFO",
    "by_stock_lag1": "Stock lag-1",
    "heuristic_deficit": "Deficit heuristico",
    "model_prioritized": "Modelo ML",
    "by_costo_quiebre": "Costo quiebre",
    "by_tasa_base": "Tasa base historica",
}


def _sidebar(backtest_df: pd.DataFrame | None) -> None:
    st.sidebar.title("Herdez Smart-Supply")
    st.sidebar.caption("Prototipo: modelo predictivo de quiebres + agente IA")

    if backtest_df is not None:
        means = backtest_df.drop(columns=["fold"]).mean()
        best = means.min()
        model_cost = means["model_prioritized"]
        inaction_cost = means["inaction"]

        st.sidebar.markdown("---")
        st.sidebar.metric(
            "Costo modelo (avg/fold)",
            f"${model_cost:,.0f}",
        )
        st.sidebar.metric(
            "Mejor baseline",
            f"${best:,.0f}",
            delta=f"{(model_cost - best) / best * 100:+.1f}%",
            delta_color="inverse",
        )
        st.sidebar.metric(
            "Ahorro vs inaccion",
            f"${inaction_cost - model_cost:,.0f}",
            delta=f"{(inaction_cost - model_cost) / inaction_cost * 100:.0f}%",
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Stack:** LightGBM + LangGraph + Gemini 2.5 Flash\n\n"
        "**Restriccion:** N=3 transferencias/dia/CEDI\n\n"
        "**Umbral economico:** p > 0.0187 (ratio 53:1)"
    )


def _tab_backtest(backtest_df: pd.DataFrame | None) -> None:
    st.header("Backtest: 7 estrategias bajo N=3")

    if backtest_df is None:
        st.warning(
            "No se encontro data/backtest_results.parquet. "
            "Ejecuta: python -m scripts.precompute_backtest"
        )
        return

    strategies = [c for c in backtest_df.columns if c != "fold"]
    means = backtest_df[strategies].mean()
    means_sorted = means.sort_values()

    summary_rows = []
    fifo_cost = means.get("fifo", 1)
    for name in means_sorted.index:
        cost = means_sorted[name]
        vs_fifo = fifo_cost - cost
        summary_rows.append(
            {
                "Estrategia": STRATEGY_LABELS.get(name, name),
                "Costo promedio/fold": f"${cost:,.0f}",
                "vs FIFO": (
                    f"+${vs_fifo:,.0f}" if vs_fifo >= 0
                    else f"-${-vs_fifo:,.0f}"
                ),
                "Ranking": len(summary_rows) + 1,
            }
        )

    st.dataframe(
        pd.DataFrame(summary_rows),
        use_container_width=True,
        hide_index=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Costo por estrategia")
        chart_data = pd.DataFrame(
            {
                "Estrategia": [
                    STRATEGY_LABELS.get(n, n) for n in means_sorted.index
                ],
                "Costo promedio ($)": means_sorted.values,
            }
        )
        st.bar_chart(chart_data, x="Estrategia", y="Costo promedio ($)")

    with col2:
        st.subheader("Costo por fold")
        fold_data = backtest_df.set_index("fold")[strategies].rename(
            columns=STRATEGY_LABELS
        )
        st.line_chart(fold_data)

    threshold_path = FIGURES_DIR / "threshold_sweep.png"
    if threshold_path.exists():
        st.subheader("Sweep de umbrales")
        st.image(str(threshold_path), use_container_width=True)

    st.markdown(
        """
        **Conclusion:** Bajo capacidad ilimitada, transferir siempre
        (all_positive) es optimo. Bajo N=3 transferencias/dia/CEDI,
        la priorizacion importa. El modelo ML queda 3ro de 7 baselines;
        su ventaja es integrar probabilidades calibradas con costos,
        algo que los baselines estaticos no hacen. Con mas datos
        (12+ meses), el modelo deberia mejorar; las reglas no.
        """
    )


def _tab_alertas(predictions_df: pd.DataFrame | None) -> None:
    st.header("Alertas diarias del agente")

    if predictions_df is None:
        st.warning(
            "No se encontro data/daily_predictions.parquet. "
            "Ejecuta: python -m scripts.precompute_backtest"
        )
        return

    col1, col2 = st.columns(2)
    folds = sorted(predictions_df["fold"].unique())
    with col1:
        fold = st.selectbox("Fold", folds, index=len(folds) - 1)

    fold_data = predictions_df[predictions_df["fold"] == fold]
    dates = sorted(fold_data["fecha"].unique())
    with col2:
        date = st.selectbox("Fecha", dates, index=0)

    day_data = fold_data[fold_data["fecha"] == date].copy()

    n_aprobadas = (day_data["decision"] == "aprobada").sum()
    n_diferidas = (day_data["decision"] == "diferida").sum()
    n_sin_alerta = (day_data["decision"] == "sin_alerta").sum()
    n_quiebres = day_data["y_true"].sum()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Transferencias", n_aprobadas)
    m2.metric("Diferidas", n_diferidas)
    m3.metric("Sin alerta", n_sin_alerta)
    m4.metric("Quiebres reales", int(n_quiebres))

    display_cols = [
        "sku_id",
        "cedi",
        "p_quiebre",
        "y_true",
        "stock_lag_1",
        "decision",
        "cedi_origen",
        "unidades",
        "costo_transferencia",
    ]
    day_display = day_data[display_cols].copy()
    day_display["p_quiebre"] = day_display["p_quiebre"].apply(
        lambda x: f"{x:.3f}"
    )

    def _color_decision(val: str) -> str:
        colors = {
            "aprobada": "background-color: #d4edda",
            "diferida": "background-color: #fff3cd",
            "sin_alerta": "background-color: #f8f9fa",
        }
        return colors.get(val, "")

    st.dataframe(
        day_display.style.map(_color_decision, subset=["decision"]),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Capacidad utilizada por CEDI origen")
    if n_aprobadas > 0:
        capacity = (
            day_data[day_data["decision"] == "aprobada"]
            .groupby("cedi_origen")
            .size()
            .reset_index(name="transferencias")
        )
        capacity["restante"] = 3 - capacity["transferencias"]
        st.dataframe(capacity, use_container_width=True, hide_index=True)
    else:
        st.info("Sin transferencias aprobadas este dia.")


def _run_agent_for_day(
    predictions_df: pd.DataFrame, fold: int, fecha: str
) -> dict | None:
    """Ejecuta el grafo del agente sobre un dia del backtest."""
    from src.agent.graph import build_graph
    from src.agent.tools import set_day_context

    day = predictions_df[
        (predictions_df["fold"] == fold)
        & (predictions_df["fecha"] == fecha)
    ]
    if day.empty:
        return None

    df_day = day.copy().reset_index(drop=True)
    y_proba = df_day["p_quiebre"].values.astype(float)

    rename_map = {"costo_quiebre_diario": "costo_quiebre_stock_diario"}
    for old, new in rename_map.items():
        if old in df_day.columns and new not in df_day.columns:
            df_day[new] = df_day[old]

    if "costo_transferencia_unidad" not in df_day.columns:
        df_day["costo_transferencia_unidad"] = 10.0
    if "lead_time_dias" not in df_day.columns:
        df_day["lead_time_dias"] = 5
    if "stock_actual" not in df_day.columns:
        df_day["stock_actual"] = df_day["stock_lag_1"]
    if "quiebre_proyectado" not in df_day.columns:
        df_day["quiebre_proyectado"] = df_day["y_true"]

    set_day_context(df_day, y_proba)
    graph = build_graph()

    state = {**_INITIAL_STATE, "fecha": str(fecha)}
    result = graph.invoke(state)
    return result


def _tab_chat(predictions_df: pd.DataFrame | None) -> None:
    st.header("Chat con el agente")

    if predictions_df is None:
        st.warning(
            "No se encontro data/daily_predictions.parquet. "
            "Ejecuta: python -m scripts.precompute_backtest"
        )
        return

    from src.data.config import get_settings

    settings = get_settings()
    if settings.google_api_key:
        st.success("Gemini 2.5 Flash conectado")
    else:
        st.info(
            "Modo demo (sin GOOGLE_API_KEY). "
            "Agrega GOOGLE_API_KEY al .env para respuestas de Gemini."
        )

    col1, col2 = st.columns(2)
    folds = sorted(predictions_df["fold"].unique())
    with col1:
        chat_fold = st.selectbox(
            "Fold", folds, index=len(folds) - 1, key="chat_fold"
        )
    fold_data = predictions_df[predictions_df["fold"] == chat_fold]
    dates = sorted(fold_data["fecha"].unique())
    with col2:
        chat_date = st.selectbox(
            "Fecha", dates, index=0, key="chat_date"
        )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Pregunta al agente sobre las alertas del dia...")

    if prompt:
        st.session_state.chat_history.append(
            {"role": "user", "content": prompt}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Ejecutando agente..."):
                    result = _run_agent_for_day(
                        predictions_df, chat_fold, chat_date
                    )
            except Exception:
                logger.exception("Error ejecutando agente")
                result = None

            if result is None:
                response = "No hay datos para esta fecha."
            else:
                n_transfers = len(result.get("transfers", []))
                n_deferred = len(result.get("deferred", []))
                explanation = result.get("explanation", "")

                response = (
                    f"**Fecha:** {chat_date} (Fold {chat_fold})\n\n"
                    f"**Transferencias:** {n_transfers} | "
                    f"**Diferidas:** {n_deferred}\n\n"
                    f"---\n\n{explanation}"
                )

            st.markdown(response)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response}
            )


def main() -> None:
    st.set_page_config(
        page_title="Herdez Smart-Supply",
        page_icon="📦",
        layout="wide",
    )

    backtest_df = load_backtest()
    predictions_df = load_predictions()

    _sidebar(backtest_df)

    tab_backtest, tab_alertas, tab_chat = st.tabs(
        ["Backtest", "Alertas", "Chat"]
    )

    with tab_backtest:
        _tab_backtest(backtest_df)

    with tab_alertas:
        _tab_alertas(predictions_df)

    with tab_chat:
        _tab_chat(predictions_df)


if __name__ == "__main__":
    main()
