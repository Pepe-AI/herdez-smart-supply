# Herdez Smart-Supply

Prototipo de 5 días: modelo predictivo de quiebres de stock + Agente de IA
con LangGraph que recomienda acciones correctivas basadas en costos.

## Stack
- Python 3.11+ con `uv` para deps
- DuckDB (storage local; sintaxis compatible con BigQuery)
- LightGBM (NO XGBoost: dataset es pequeño, ~1200 filas)
- LangGraph (NO LangChain plano: el manejo de estado del agente se evalúa)
- Gemini 2.5 Flash vía API gratuita (`langchain-google-genai`)
- Streamlit para la UI
- `pytest`, `ruff`, `mypy --strict` en `src/economics` y `src/ml`

## Comandos

```bash
uv sync                              # instalar deps
python -m src.data.ingest            # Excel → DuckDB
python -m src.ml.train               # entrenar y guardar modelo
python -m src.ml.evaluate            # métricas con TimeSeriesSplit
python -m scripts.precompute_backtest # generar parquets para dashboard
streamlit run src/app/main.py        # dashboard (3 tabs: Backtest, Alertas, Chat)
pytest tests/ -v                     # tests (46 total)
ruff check src/ tests/ && ruff format src/ tests/
```

## Hechos del dataset (NO inventar señal que no existe)

Señal de features categóricas (verificado con tests formales):
- `Promocion_Activa`: sin efecto en ventas (Welch p=0.47) pero reduce tasa
  de quiebre en 4 de 5 SKUs (hasta -21pp). Incluir como feature; dejar que
  el modelo decida su importancia.
- `Clima`: sin señal ni en ventas (Kruskal p=0.72) ni en target (Chi² p=0.93).
  Mantener como placeholder.

Estas variables son atributos, no features dinámicas:
- `Lead_Time_Dias`: constante por CEDI (Norte=3, resto=5).
- `Costo_Quiebre_Stock_Diario`: constante por SKU ($15k salsas, $8k resto).

La única variable de costo que varía día a día es `Costo_Transferencia_Unidad`.

Stock = 0 NUNCA ocurre en el histórico (mínimo observado: 50 unidades).
Por eso el target es un proxy proyectado, no una observación directa.

## Leakage resuelto: estrategia de features lagged

El target `(stock_actual - ventas_rolling_7d * 5) < 0` es determinista
sobre `stock_actual` y `ventas_unidades`. Si se usan como features,
el modelo aprende la fórmula (AUC-PR ≈ 1.0) en lugar de patrones reales.

**Solución implementada:** mantener el target proxy (interpretable) pero
usar solo features lagged (t-1 o anterior): `stock_lag_{1,3,5,7}`,
`ventas_lag_{1,3,5,7}`, `ventas_rolling_7d_lag1`, ratios de cobertura
lagged. `stock_actual` y `ventas_unidades` están en EXCLUDE_COLS.

AUC-PR honesto con esta estrategia: ~0.45 ± 0.06 (TimeSeriesSplit, 5 folds).
Baseline random ≈ 0.335 (prevalencia del target).

## Hallazgos críticos del Día 1

### Ratio real de costos: 53:1

- FN promedio: $53,333 MXN. Cálculo: mix ponderado de 5 SKUs ×
  costo_quiebre × 5 días exposición. Salsas: $15k×5=$75k.
  Atún/Champiñones/Mole: $8k×5=$40k. Promedio: ($75k×2+$40k×3)/5=$54k.
  (Fuente: `scripts/cost_baselines.py`)
- FP promedio: $997 MXN. Cálculo: costo_transferencia_unidad (~$10.45) ×
  max(deficit_estimado, 50 unidades mínimas). Promedio ponderado de
  all_positive: $175,423 FP / 176 transfers.
  (Fuente: `scripts/cost_baselines.py`)
- Ratio: $53,333 / $997 = **53:1**.

### Umbral económico óptimo: 0.0187

Transferir si: `p_quiebre > costo_FP / (costo_FP + costo_FN)`
= $997 / ($997 + $53,333) = 0.0187.
(Fuente: `scripts/threshold_sweep.py`)

### Hallazgo central: all_positive es Pareto-óptimo bajo capacidad ilimitada

Con ratio 53:1, no existe umbral del modelo que supere a transferir
siempre. La curva costo-vs-umbral es monótonamente creciente (sweep
de umbrales 0.01 a 0.50, `scripts/threshold_sweep.py`). Esto es
propiedad del régimen de costos, no debilidad del modelo.

### Decisión arquitectónica

El modelo se mantiene como sensor de probabilidades diarias. Su valor
económico emerge bajo restricciones logísticas (capacidad limitada),
donde priorizar alertas por costo esperado supera al orden arbitrario.
Simulación con restricciones: pendiente Día 2 (`costs_v2.py`).

### Parámetro de diseño del agente: N=3 transferencias/día por CEDI

Justificación: cada transferencia requiere ~2h (coordinación + picking +
carga + documentación). Jornada de 8h con overlap parcial → 3 simultáneas
máximo. Con 4 CEDIs → 12 transferencias/día en la red.
(Detalle: `docs/architecture.md`)

### Bugs detectados en meta-auditoría (Día 2)

**Bug 1: asimetría en selección de origen.** El script inline de la
primera auditoría usaba "primer origen válido" (iterrows + break) para
los baselines mientras `costs_v2.select_origin()` usa "mejor origen"
(mayor excedente) para el modelo. Corregido: todas las estrategias
usan `select_origin()` de `costs_v2.py`.

**Bug 2: leakage temporal en by_stock_bajo.** Usaba `stock_actual`
(stock post-ventas del día) para priorizar al inicio del día. Eso es
información del futuro. Corregido a `stock_lag_1` (stock de ayer).

**Bug 3: divergencia silenciosa entre backtest y agente (Día 3).**
`simulate_day_with_priority()` divergía del agente en 3 aspectos:
(1) scoring con costo_esperado puro en vez de beneficio_neto,
(2) sin filtro de umbral económico (p > 0.0187),
(3) sin filtro de beneficio_neto > 0.
Corregido: `y_proba` es obligatorio, los dos filtros económicos se
aplican uniformemente a las 7 estrategias antes de priorizar.

**Impacto en ranking bajo N=3:**
- Modelo: 5° → 3° de 6 baselines (mejora dos posiciones)
- by_stock_bajo: 1° → 5° (pierde cuatro posiciones, su ventaja era leakage)

**Números headline verificados (producción, filtros uniformes):**
- Costo del modelo bajo N=3: $1,046,657/fold (producción, costs_v2.py)
- Mejor baseline (tasa_base): $912,709/fold
- Brecha modelo vs mejor baseline: $133,948/fold (+14.7%)
- Posición: 3° de 7 baselines honestos
- Ranking:
  tasa_base > costo_quiebre > modelo > heuristic_deficit > stock_lag1 > fifo > inacción
  (Fuente: `costs_v2.run_capacity_comparison`, `data/backtest_results.parquet`)

## Hallazgos del Día 3

### Dashboard Streamlit funcional (3 tabs)

- **Backtest:** tabla de 7 estrategias, gráficos comparativos, threshold sweep
- **Alertas:** explorador por fold/fecha, tabla con decisiones coloreadas,
  capacidad por CEDI
- **Chat:** integración con agente LangGraph, Gemini 2.5 Flash (o mock)

### 7 estrategias portadas a código de producción

`simulate_day_with_priority()` unifica la lógica de todas las estrategias:
misma función, mismo `select_origin()`, mismos filtros económicos
(p > 0.0187 + beneficio_neto > 0), solo cambian los scores de prioridad.
`y_proba` es obligatorio (no opcional). Esto corrige permanentemente
los 3 bugs de las auditorías.

### Integración Gemini: auto-detección

`get_llm()` en `graph.py` lee `GOOGLE_API_KEY` de `.env`:
- Con key: `ChatGoogleGenerativeAI(model="gemini-2.5-flash")`
- Sin key: `_MockLLM` que aprueba el plan determinista
- Test de equivalencia decisional (skip si no hay key): verifica que
  los nodos deterministas producen transfers idénticos con ambos LLMs

### Tests: 48 total (47 pass, 1 skip)

- 20 tests de costs_v2 (incluye equivalencia backtest-agente + stock_lag_1 vs stock_actual)
- 7 tests de agente (incluye equivalencia Gemini, skip si no hay key)
- 14 tests de features
- 7 tests de economics

## Definiciones de negocio

```python
# Target del modelo ML
quiebre_proyectado = (stock_actual - ventas_promedio_7d * 5) < 0

# Lógica de decisión del agente
costo_no_actuar  = costo_quiebre_diario * p_quiebre * dias_expuestos
costo_transferir = costo_transferencia_unidad * unidades_movidas
# Transferir si costo_transferir < costo_no_actuar AND origen no entra en riesgo
# Umbral económico: p > 0.0187 (equivalente al ratio de costos 53:1)
```

Ventana = 5 días (coincide con lead time del CEDI más lento; alerta accionable).

## Scope de verificación de tipos

- `mypy --strict` aplica a: `src/economics/`, `src/ml/`
- `src/agent/` queda fuera: LangGraph usa signatures dinámicas
  (StateGraph, tool_calls) que generan falsos positivos con strict.
  Configurado en `pyproject.toml` como override con `strict = false`.
- `src/data/`, `src/app/` también fuera (I/O, Streamlit).
- Comando de validación: `mypy src/economics/ src/ml/`

## Convenciones

- Idioma: código y nombres en inglés; comentarios, prompts del agente,
  textos de UI y docs en español. La presentación es para Herdez México.
- Validación temporal SIEMPRE (`TimeSeriesSplit`); nunca random split.
- `logging`, no `print`. Config con `pydantic-settings` desde `.env`.
- Type hints obligatorios en funciones públicas.
- Métricas técnicas: AUC-PR (no AUC-ROC), precision/recall del quiebre,
  matriz de confusión monetizada en MXN. Métricas de negocio (deben
  aparecer en el dashboard): fill rate, ahorro logístico en MXN, ROI vs
  costo estimado del sistema.
- Commits convencionales (feat/fix/docs/refactor/test/chore).

## Layout

```
src/
├── data/       ingesta + features (DuckDB queries)
├── ml/         train, predict, evaluate
├── economics/  cálculos de costo puros (con tests)
├── agent/      LangGraph: graph.py, tools.py, state.py, prompts.py
└── app/        Streamlit (tabs: Alertas, Chat, Backtest)
tests/          tests de economics y agent/tools
docs/           architecture.md, presentation/
```
