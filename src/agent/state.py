"""Estado del agente LangGraph para decisiones de transferencia."""

from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Estado que fluye por el grafo del agente económico.

    Nodos deterministas (1-3) escriben alerts/transfers/deferred.
    Nodos LLM (4-5) leen el plan y generan explicación.
    """

    # Historial de mensajes (conversación con Gemini)
    messages: Annotated[list, add_messages]

    # Fecha de simulación (YYYY-MM-DD)
    fecha: str

    # Alertas generadas por el modelo (nodo 1)
    alerts: list[dict]

    # Alertas evaluadas con costo-beneficio positivo (nodo 2)
    evaluated_alerts: list[dict]

    # Transferencias aprobadas (nodo 3)
    transfers: list[dict]

    # Alertas diferidas por capacidad o beneficio negativo (nodo 3)
    deferred: list[dict]

    # Capacidad usada por CEDI origen {cedi: n_usados}
    capacity_used: dict[str, int]

    # Resumen ejecutivo en español (nodo 5)
    explanation: str
