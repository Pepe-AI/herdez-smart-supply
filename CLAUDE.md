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
streamlit run src/app/main.py        # dashboard
pytest tests/ -v                     # tests
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

## Definiciones de negocio

```python
# Target del modelo ML
quiebre_proyectado = (stock_actual - ventas_promedio_7d * 5) < 0

# Lógica de decisión del agente
costo_no_actuar  = costo_quiebre_diario * p_quiebre * dias_expuestos
costo_transferir = costo_transferencia_unidad * unidades_movidas
# Transferir si costo_transferir < costo_no_actuar AND origen no entra en riesgo
```

Ventana = 5 días (coincide con lead time del CEDI más lento; alerta accionable).

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
