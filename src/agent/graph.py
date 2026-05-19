"""Grafo LangGraph del agente económico de Supply Chain.

Estructura:
    [START] → [generate_alerts] → [evaluate_costs] → [prioritize]
           → [decide] ⇄ [tools] → [explain] → [END]

Nodos 1-3 son deterministas (Python puro, sin LLM).
Nodos 4-5 usan Gemini para revisión y explicación en español.
"""

import json
import logging
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from src.agent.prompts import SYSTEM_PROMPT
from src.agent.state import AgentState
from src.agent.tools import _config, _day_data, get_tools
from src.economics.costs_v2 import generate_alerts, prioritize_and_allocate

logger = logging.getLogger(__name__)


# --- Nodo 1: Generar alertas (determinista) ---
def generate_alerts_node(state: AgentState) -> dict:
    """Ejecuta el modelo y filtra alertas por umbral económico."""
    df_day = _day_data["df_day"]
    y_proba = _day_data["y_proba"]

    alerts = generate_alerts(df_day, y_proba, _config)
    logger.info("Alertas generadas: %d", len(alerts))
    return {"alerts": alerts}


# --- Nodo 2: Evaluar costos (determinista) ---
def evaluate_costs_node(state: AgentState) -> dict:
    """Filtra alertas con beneficio neto positivo."""
    alerts = state["alerts"]
    evaluated = [a for a in alerts if a["beneficio_neto"] > 0]
    logger.info("Alertas con beneficio positivo: %d de %d", len(evaluated), len(alerts))
    return {"evaluated_alerts": evaluated}


# --- Nodo 3: Priorizar y asignar (determinista) ---
def prioritize_node(state: AgentState) -> dict:
    """Asignación greedy con restricción de capacidad N=3/CEDI."""
    df_day = _day_data["df_day"]
    alerts = state["evaluated_alerts"]

    approved, deferred = prioritize_and_allocate(alerts, df_day, _config)

    # Convertir TransferProposal a dict para serialización
    transfers = [
        {
            "sku_id": t.sku_id,
            "cedi_origen": t.cedi_origen,
            "cedi_destino": t.cedi_destino,
            "p_quiebre": t.p_quiebre,
            "beneficio_neto": t.beneficio_neto,
            "unidades": t.unidades,
            "costo_transferencia": t.costo_transferencia,
        }
        for t in approved
    ]

    # Rastrear capacidad usada
    capacity_used: dict[str, int] = {}
    for t in approved:
        capacity_used[t.cedi_origen] = capacity_used.get(t.cedi_origen, 0) + 1

    logger.info(
        "Transferencias aprobadas: %d, diferidas: %d", len(transfers), len(deferred)
    )
    return {
        "transfers": transfers,
        "deferred": deferred,
        "capacity_used": capacity_used,
    }


# --- Nodo 4: Decidir (LLM revisa el plan) ---
def decide_node(state: AgentState, llm) -> dict:
    """Gemini revisa el plan y puede ajustar vía tools."""
    transfers = state.get("transfers", [])
    deferred = state.get("deferred", [])

    summary = (
        f"Plan propuesto:\n"
        f"- {len(transfers)} transferencias aprobadas\n"
        f"- {len(deferred)} alertas diferidas\n\n"
        f"Transferencias:\n{json.dumps(transfers, indent=2, ensure_ascii=False)}\n\n"
        f"Diferidas:\n{json.dumps(deferred, indent=2, ensure_ascii=False)}"
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=summary),
    ]

    response = llm.invoke(messages)
    return {"messages": [response]}


# --- Nodo 5: Explicar (LLM genera resumen en español) ---
def explain_node(state: AgentState, llm) -> dict:
    """Genera resumen ejecutivo en español para el Director."""
    transfers = state.get("transfers", [])
    deferred = state.get("deferred", [])

    # Si el LLM ya generó una respuesta en decide, usarla
    messages = state.get("messages", [])
    if messages:
        last = messages[-1]
        if hasattr(last, "content") and last.content:
            return {"explanation": last.content}

    # Fallback: pedir explicación explícita
    prompt = (
        f"Genera un resumen ejecutivo en español de estas decisiones:\n"
        f"Transferencias: {json.dumps(transfers, ensure_ascii=False)}\n"
        f"Diferidas: {json.dumps(deferred, ensure_ascii=False)}"
    )
    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    return {"explanation": response.content}


# --- Edge condicional ---
def should_use_tools(state: AgentState) -> Literal["tools", "explain"]:
    """Si Gemini pidió llamar tools, ir a tools; si no, a explain."""
    messages = state.get("messages", [])
    if messages:
        last = messages[-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
    return "explain"


# --- Constructor del grafo ---
def build_graph(llm=None, config=None):
    """Construye el grafo del agente económico.

    Args:
        llm: ChatModel (Gemini o mock). Si None, usa mock.
        config: CapacityConfig. Si None, usa default.
    """
    if llm is None:
        # Mock LLM que aprueba el plan determinista sin cambios
        class _MockLLM:
            def invoke(self, messages):
                return AIMessage(
                    content="Plan aprobado. Resumen: sin cambios al plan determinista."
                )

            def bind_tools(self, tools):
                return self

        llm = _MockLLM()

    tools = get_tools()
    llm_with_tools = llm.bind_tools(tools) if hasattr(llm, "bind_tools") else llm

    # Funciones con LLM inyectado via closure
    def _decide(state: AgentState) -> dict:
        return decide_node(state, llm_with_tools)

    def _explain(state: AgentState) -> dict:
        return explain_node(state, llm_with_tools)

    def _tool_node(state: dict) -> dict:
        """Ejecuta tool calls del LLM."""
        from langchain_core.messages import ToolMessage

        result = []
        tools_by_name = {t.name: t for t in tools}
        for tool_call in state["messages"][-1].tool_calls:
            tool_fn = tools_by_name[tool_call["name"]]
            observation = tool_fn.invoke(tool_call["args"])
            result.append(
                ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
            )
        return {"messages": result}

    # Construir grafo
    graph = StateGraph(AgentState)

    graph.add_node("generate_alerts", generate_alerts_node)
    graph.add_node("evaluate_costs", evaluate_costs_node)
    graph.add_node("prioritize", prioritize_node)
    graph.add_node("decide", _decide)
    graph.add_node("tools", _tool_node)
    graph.add_node("explain", _explain)

    graph.add_edge(START, "generate_alerts")
    graph.add_edge("generate_alerts", "evaluate_costs")
    graph.add_edge("evaluate_costs", "prioritize")
    graph.add_edge("prioritize", "decide")
    graph.add_conditional_edges("decide", should_use_tools, ["tools", "explain"])
    graph.add_edge("tools", "decide")
    graph.add_edge("explain", END)

    return graph.compile()
