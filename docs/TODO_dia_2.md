# TODO Dia 2 — Agente economico + simulacion con restricciones

## Prioridad 1: simulacion con capacidad limitada

1. **`src/economics/costs_v2.py`**: nueva simulacion con restriccion de
   capacidad N=3 transferencias/dia por CEDI origen. Re-evaluar las 5
   estrategias (inaccion, all_positive, heuristica, tasa_base, modelo)
   bajo esta restriccion con TimeSeriesSplit 5 folds.

2. **Numero headline real**: % de ahorro capturado por modelo + agente vs
   `all_positive_truncado_por_capacidad`. Este es EL numero que va a la
   presentacion. No existe todavia — no usar proyecciones.

## Prioridad 2: implementar agente con LangGraph

3. **`src/agent/`**: implementar nodos del flujo definido en
   `docs/architecture.md`. Usar Gemini 2.5 Flash para razonamiento.
   Archivos: `state.py`, `tools.py`, `prompts.py`, `graph.py`.

4. **Validacion de restriccion de stock origen**: implementar como nodo
   del agente, no como filtro previo. El agente decide si el origen
   queda en riesgo.

## Prioridad 3: tests

5. **`tests/test_costs_v2.py`**: tests de la simulacion con capacidad.
6. **`tests/test_agent_decisions.py`**: tests de la logica de decision
   del agente (sin Gemini, solo logica economica).

## Decision pendiente

7. **Umbral de seguridad del stock origen**: se necesita definir cuanto
   stock debe retener el CEDI origen tras transferir. Sugerencia inicial:
   percentil 25 del stock historico del grupo. A validar empiricamente
   en la simulacion del punto 1.
