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

El EDA mostró que estas variables NO tienen señal predictiva en los datos:
- `Promocion_Activa`: ventas con promo (137.5) ≤ sin promo (140.5).
- `Clima`: diferencias entre Despejado/Lluvia/Tormenta < 3%.

Mantenerlas en el pipeline como placeholders, pero NO presentarlas como
features importantes. Si el modelo les asigna alta importancia, revisar.

Estas variables son atributos, no features dinámicas:
- `Lead_Time_Dias`: constante por CEDI (Norte=3, resto=5).
- `Costo_Quiebre_Stock_Diario`: constante por SKU ($15k salsas, $8k resto).

La única variable de costo que varía día a día es `Costo_Transferencia_Unidad`.

Stock = 0 NUNCA ocurre en el histórico (mínimo observado: 50 unidades).
Por eso el target es un proxy proyectado, no una observación directa.

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
