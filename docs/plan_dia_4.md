# Plan del Dia 4 — Contexto, Inventario y Brechas

## Tarea 1 — Resumen de archivos leidos

| # | Archivo | Resumen |
|---|---------|---------|
| 1 | `CLAUDE.md` | Documento maestro: stack, hechos del dataset, leakage resuelto, hallazgos Dia 1 (ratio 53:1, umbral 0.0187, all_positive Pareto), 3 bugs documentados, numeros headline produccion ($1,046,657/$912,709/+14.7%), Dia 3 (dashboard, 7 estrategias, Gemini, 48 tests). |
| 2 | `docs/architecture.md` | Diseno del agente economico: restriccion N=3, flujo de 5 pasos (generar alertas, evaluar costos, priorizar, decidir, explicar), tabla de valor por componente, politica de filtros uniforme del backtest (p > 0.0187 + beneficio_neto > 0). |
| 3 | `docs/meta_auditoria_dia2.md` | Verificacion de la auditoria original: Bug 1 (asimetria select_origin), Bug 2 (leakage stock_actual en by_stock_bajo), Bug 3 (divergencia backtest-agente), ranking corregido, tabla produccion vs meta-auditoria, veredicto final. |
| 4 | `docs/TODO_dia_4.md` | Lista de pendientes: Prioridad 0 (actualizar Word), P1 (fill rate/ROI, mejorar chat, activar Gemini), P2 (tests faltantes), P3 (screenshots, narrativa presentacion), P4 (deploy opcional). Decision resuelta: usar numeros de produccion. |
| 5 | `notebooks/01_eda.ipynb` | EDA completo: Promocion (Welch p=0.47 ventas, reduce quiebre en 4/5 SKUs), Clima (sin senal p=0.93), constantes verificadas, stock minimo=50, leakage demostrado (AUC-PR leaked=0.99 vs lagged=0.48 vs exogenas=0.45), sweep de umbrales, conclusion Dia 1: all_positive Pareto-optimo sin restricciones, valor emerge con N=3. |
| 6 | `src/economics/costs_v2.py` | Modulo puro con 7 funciones: `CapacityConfig`, `generate_alerts`, `compute_origin_safety`, `select_origin`, `prioritize_and_allocate`, `simulate_day`/`simulate_day_with_priority` (y_proba obligatorio, filtros uniformes), `run_capacity_comparison` con 7 estrategias y docstring corregido. |
| 7 | `src/agent/graph.py` | Grafo LangGraph de 6 nodos: 3 deterministas (generate_alerts, evaluate_costs, prioritize) + 2 con LLM (decide, explain) + tools. `get_llm()` con auto-deteccion. `_MockLLM` como fallback. Edge condicional para tool calls. |
| 8 | `src/app/main.py` | Dashboard Streamlit: 3 tabs (Backtest, Alertas, Chat), sidebar con metricas dinamicas del parquet, `@st.cache_data`, `_run_agent_for_day()` invoca el grafo LangGraph completo. |

---

## Respuestas a las preguntas de contexto

### Numero headline definitivo

- **Modelo:** $1,046,657/fold (3ro de 7)
- **Mejor baseline (tasa_base):** $912,709/fold
- **Brecha:** +$133,948/fold (+14.7%)

### Ahorro total del sistema vs inaccion

- **$2,587,343/fold** (71% de reduccion vs inaccion de $3,634,000/fold)

### Tres bugs documentados y correcciones

| Bug | Problema | Correccion |
|-----|----------|------------|
| Bug 1: asimetria select_origin | Baselines usaban "primer valido" vs modelo usaba "mejor excedente" | `select_origin()` uniforme para todas las estrategias |
| Bug 2: leakage temporal | by_stock_bajo usaba `stock_actual` (info del futuro) para priorizar | Cambiado a `stock_lag_1` (stock de ayer) |
| Bug 3: divergencia backtest-agente | Scoring con costo_esperado puro, sin filtro umbral, sin filtro beneficio_neto | `y_proba` obligatorio, 2 filtros uniformes para las 7 estrategias |

### Politica de filtros uniforme del backtest

Todas las 6 estrategias activas aplican 2 filtros antes de priorizar:

1. **Filtro de umbral economico:** p > 0.0187 (derivado del regimen de costos 53:1)
2. **Filtro de beneficio neto positivo:** costo_esperado_no_actuar > costo_transferencia

La unica diferencia entre estrategias es el criterio de ordenamiento
del conjunto ya filtrado. Esto garantiza que ninguna estrategia tome
decisiones economicamente irracionales.

### Decision arquitectonica sobre el rol del LLM

El LLM (Gemini 2.5 Flash) tiene rol de **revision y explicacion**, NO
de decision. Los nodos 1-3 del grafo son deterministas (Python puro,
funciones de costs_v2.py). El LLM solo actua en nodos 4 (revisar plan)
y 5 (generar explicacion en espanol). Las decisiones economicas son
identicas con o sin LLM -- test de equivalencia decisional lo verifica.

### Entrega al final del Dia 5

**Documentacion incompleta.** No se pudo leer `docs/Prueba_Tecnica_Herde_IA.pdf`
(no hay lector PDF instalado). Los TODOs mencionan: presentacion de 60 min,
dashboard funcional, documento de arquitectura GCP, evidencia Word actualizada.
Se necesita confirmacion de que se entrega exactamente.

---

## Tarea 2 — Inventario de estado al inicio del Dia 4

### Sistema implementado (verificado con codigo)

| Componente | Estado | Verificacion |
|---|---|---|
| Pipeline ML (features lagged, AUC-PR ~0.45) | Completo | 14 tests features, notebook EDA |
| costs_v2.py con filtros uniformes | Completo | 20 tests, `y_proba` obligatorio |
| Agente LangGraph (5 nodos + tools) | Completo | 7 tests (incl. equivalencia Gemini) |
| Dashboard Streamlit (3 tabs) | Completo | Arranca sin errores, metricas dinamicas |
| Backtest precomputado | Completo | 2 parquets generados |
| 48 tests passing | Verificado | 47 pass + 1 skip (Gemini sin key) |
| ruff clean | Verificado | 0 errors |
| 3 bugs documentados con tests regresion | Completo | Tests de equivalencia + stock_lag1 |

### Pendientes del Dia 3 no cerrados

1. **CLAUDE.md linea 24** dice "46 total" en la seccion de Comandos -- deberia decir 48. (Inconsistencia menor.)
2. **meta_auditoria_dia2.md lineas 219-222** — La seccion de Bug 3 "Correccion" dice que `y_proba` es "opcional que activa filtros". Ya es falso: `y_proba` ahora es obligatorio.
3. **meta_auditoria_dia2.md lineas 238-248** — La tabla "Numeros de produccion actualizados" muestra numeros pre-filtros-uniformes (tasa_base $915,470 vs produccion actual $912,709).
4. **.env.example** borrado (aparece como `deleted` en git status) — si era util para onboarding, quizas recrear.
5. **evidencia_ml_herdez.docx** no esta trackeado en git.

Ninguno bloquea el Dia 4, pero los items 1-3 deberian corregirse en el commit de cierre.

---

## Tarea 3 — Brechas identificadas

### Entrega A: Documento de arquitectura GCP

| Lo que debe existir | Lo que existe hoy | Brecha |
|---|---|---|
| Diagrama de servicios GCP (Mermaid/ASCII) | Nada | Crear desde cero |
| Justificacion pieza por pieza (BigQuery, Vertex AI, Cloud Run, CI/CD) | Solo mencion DuckDB "compatible con BigQuery" en CLAUDE.md | Crear desde cero |
| Mapeo prototipo → produccion | Nada | Crear desde cero |
| Estimacion cualitativa de costos GCP | Nada | Crear desde cero |
| Archivo `docs/architecture_gcp.md` | No existe | Crear |

### Entrega B: Esqueleto de presentacion

| Lo que debe existir | Lo que existe hoy | Brecha |
|---|---|---|
| Estructura de 60 minutos | Nada | Crear desde cero |
| Narrativa Gerente IA (~30 min) | Nada (solo bullets en TODO_dia_4.md) | Crear |
| Narrativa Director SC (~30 min) | Nada | Crear |
| Mapping slide → evidencia del repo | Nada | Crear |
| Preguntas hostiles + respuestas | Nada | Crear |
| Archivo `docs/presentation/outline.md` | Directorio no existe | Crear |

**Dependencias clave:** Los numeros headline ya estan verificados. El EDA
del notebook tiene las tablas y graficos. El dashboard existe como demo.
Lo que falta es la estructura narrativa y el documento GCP.

---

## Tarea 5 — Riesgos identificados

| # | Riesgo | Severidad | Mitigacion |
|---|--------|-----------|------------|
| R1 | Formato presentacion (slides vs md) | Media | Outline.md como fuente de verdad, convertible a cualquier formato |
| R2 | No hay diagrama GCP previo | Baja | Crear con Mermaid desde cero |
| R3 | Demo en vivo requiere entorno | Media | Parquets commiteados, mock fallback, solo necesita `uv sync` |
| R4 | Rate limit Gemini en demo | Baja | Mock automatico como fallback, screenshots backup |
| R5 | Word con numeros viejos | Alta | Actualizar antes de presentacion o excluir |

---

## Preguntas pendientes antes de arrancar

1. **Que dice el PDF de la prueba tecnica sobre las entregas del Dia 5?**
   No se pudo leer -- se necesita resumen de que se espera exactamente.

2. **Formato de presentacion?** (PPT/Google Slides/Markdown renderizado)
   Afecta nivel de detalle del outline.

3. **Demo en vivo del dashboard durante la presentacion?** Si/No.
