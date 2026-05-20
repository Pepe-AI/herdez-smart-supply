# TODO Dia 3 — Dashboard Streamlit + Integracion Gemini

## Bloque 1A: portar 6 estrategias a costs_v2.py (~1h)

1. **`src/economics/costs_v2.py`**: agregar `simulate_day_with_priority()`
   que unifique la logica de las 6 estrategias (inaction, fifo,
   by_costo_quiebre, by_stock_lag1, by_tasa_base, heuristic_deficit,
   model_prioritized). Todas usan `select_origin()` para igualdad.
   Eliminar `_simulate_strategy_random_truncated` (reemplazado por fifo).

2. **`tests/test_costs_v2.py`**: test de regresion para las 6 estrategias
   con datos sinteticos. Test de que by_stock_lag1 usa stock_lag_1.

## Bloque 1B: script de precomputacion (~30min)

3. **`scripts/precompute_backtest.py`**: genera
   `data/backtest_results.parquet` y `data/daily_predictions.parquet`
   para que Streamlit los lea sin reentrenar.

## Bloque 2: Streamlit base + Tab Backtest (~1.5h)

4. **`src/app/main.py`**: sidebar con metricas headline, 3 tabs
   (Backtest funcional, Alertas placeholder, Chat placeholder).
   Tab Backtest: tabla de 6 estrategias, grafico threshold_sweep.png.

## Bloque 3: Tab Alertas (~1.5h)

5. **`src/app/main.py`**: selector de fecha, tabla de alertas con
   p_quiebre + decisiones, capacidad por CEDI, metricas del dia.

## Bloque 4: integracion Gemini real (~1h)

6. **`src/agent/graph.py`**: si GOOGLE_API_KEY disponible, usar
   `ChatGoogleGenerativeAI(model="gemini-2.5-flash")`. Si no, mock.

7. **Test de equivalencia decisional**: verificar que mock LLM y
   Gemini real producen transfers y deferred identicos (solo cambia
   la explicacion). Marcar `@pytest.mark.skipif` si no hay key.

## Bloque 5: Tab Chat (~1.5h)

8. **`src/app/main.py`**: `st.chat_input()` + historial + invocacion
   del grafo. Sin key: mensaje informativo + output mock.

## Bloque 6: polish (~1h)

9. Manejo de errores, `@st.cache_resource`, mensajes informativos.
   Sidebar con metricas resumen.

## Bloque 7: cierre del Dia 3 (~30min)

10. Actualizar CLAUDE.md con hallazgos del Dia 3.
11. Generar TODO_dia_4.md con pendientes de presentacion + GCP.
12. Screenshots del dashboard para la presentacion.
13. Commits atomicos + push.
