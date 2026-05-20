# Preguntas hostiles anticipadas — Herdez Smart-Supply

Respuestas preparadas para la sesion de Q&A. Organizadas por
audiencia y ordenadas por probabilidad (alta primero).

---

## Preguntas tecnicas — Gerente de IA

### P1. "¿Por que el modelo pierde 14.7% contra una regla simple de priorizacion?"

**Probabilidad:** ALTA

**Respuesta:** Porque en 60 dias con 5 productos, los rankings de
riesgo son tan estables que una regla fija que memoriza el orden
("siempre atender salsas primero") funciona bien. Es una propiedad
del dataset, no del modelo. En produccion con 12+ meses, los
rankings cambian por estacionalidad (Cuaresma sube atun, Buen Fin
dispara todo), y un modelo que se reentrena se adapta — la regla no.

**Soporte:** `data/backtest_results.parquet` (modelo $1,046,657 vs
tasa_base $912,709). `notebooks/01_eda.ipynb` seccion de cierre
(~linea 1083): conclusiones sobre estabilidad de rankings.

---

### P2. "¿AUC-PR de 0.45 es aceptable?"

**Probabilidad:** ALTA

**Respuesta:** AUC-PR depende de la prevalencia del target: baseline
random con 33.5% de prevalencia da ~0.33, asi que 0.45 representa
una mejora real del modelo. Pero la metrica relevante no es AUC-PR
sino el costo total bajo restriccion N=3. El modelo es competitivo
ahi: $1.05M MXN/fold vs $3.6M de inaccion (71% de reduccion). La
precision del modelo importa menos que su capacidad de ordenar
correctamente las alertas por urgencia.

**Soporte:** `notebooks/01_eda.ipynb` seccion de leakage (AUC-PR
lagged=0.45 vs random baseline=0.335). `data/backtest_results.parquet`
(costo total por estrategia).

---

### P3. "¿60 dias son suficientes para entrenar un modelo de produccion?"

**Probabilidad:** ALTA

**Respuesta:** No. 60 dias no capturan estacionalidad, eventos
comerciales ni tendencias a mediano plazo. El prototipo valida la
arquitectura y la logica de decision, no la escala. El modelo
entrenado con 60 dias NO se deploya a produccion — la propuesta GCP
incluye reentrenamiento con 12+ meses de historico real en Vertex AI
Pipelines. El primer paso del piloto es conectar datos historicos
reales.

**Soporte:** `docs/architecture_gcp.md` seccion "Mapeo prototipo →
produccion" (fila ML Pipeline: "el modelo y la validacion se
conservan; se agrega reentrenamiento programado").
`CLAUDE.md` seccion "Hechos del dataset".

---

### P4. "¿Como se valida que no hay leakage?"

**Probabilidad:** MEDIA-ALTA

**Respuesta:** Se detectaron y corrigieron 3 bugs de leakage/
inconsistencia durante el prototipo, cada uno con test de regresion
que previene recurrencia. Validacion temporal estricta con
TimeSeriesSplit (nunca random split). Las features son exclusivamente
lagged (t-1 o anterior) — `stock_actual` y `ventas_unidades` estan
en EXCLUDE_COLS. La evidencia mas fuerte: AUC-PR bajo de 0.99
(leakeado) a 0.45 (honesto) al corregirlo.

**Soporte:** `docs/meta_auditoria_dia2.md` lineas 187–248 (Bug 3).
`tests/test_costs_v2.py` lineas 276–345 (clase
`TestBacktestMatchesAgentDecisions`). `CLAUDE.md` seccion "Leakage
resuelto".

---

### P5. "¿Como manejan cold start de nuevos SKUs sin historico?"

**Probabilidad:** ALTA

**Respuesta:** Para un SKU nuevo sin historico, el modelo no tiene
base para predecir. El fallback es la heuristica by_costo_quiebre
(que no requiere prediccion, solo el costo de quiebre del producto).
Durante los primeros 30-60 dias, el SKU nuevo opera con esta regla
mientras acumula datos. Cuando hay suficiente historico, el modelo
lo incorpora en el siguiente reentrenamiento. El monitoreo de drift
detecta cuando el modelo ya tiene suficiente senal para ese SKU.

**Soporte:** `src/economics/costs_v2.py` linea 265
(`simulate_day_with_priority`: la estrategia by_costo_quiebre no
usa y_proba para ordenar, solo costo_quiebre_diario).
`docs/architecture_gcp.md` seccion "Validacion de calidad"
(monitoreo de drift).

---

### P6. "¿Que pasa si el regimen de costos 53:1 cambia?"

**Probabilidad:** MEDIA

**Respuesta:** El umbral economico (0.0187) se deriva
matematicamente del ratio FP/FN. Si los costos cambian (por ejemplo,
nuevo contrato de transporte o cambio de precio de un SKU), el
umbral se recalcula automaticamente. La politica de filtros en
costs_v2.py es parametrica: CapacityConfig recibe los costos como
input, no como constantes. No requiere redespliegue — solo actualizar
la configuracion.

**Soporte:** `src/economics/costs_v2.py` lineas 1–30 (dataclass
`CapacityConfig` con parametros configurables).
`docs/architecture.md` lineas 111–130 (derivacion del umbral como
funcion del ratio de costos).

---

### P7. "¿Por que LightGBM y no XGBoost, RandomForest o una red neuronal?"

**Probabilidad:** MEDIA

**Respuesta:** Con ~1,200 filas, el dataset es muy chico para redes
neuronales (requieren ordenes de magnitud mas datos). XGBoost y
LightGBM son comparables en datasets pequenos; LightGBM tiene
manejo nativo de categoricas y es mas eficiente en entrenamiento.
RandomForest es menos expresivo para interacciones complejas. En
produccion con 100k+ filas, se puede hacer benchmark formal entre
los tres — la arquitectura de Vertex AI lo permite sin cambiar
nada mas.

**Soporte:** `src/ml/train.py` linea 52 (funcion `train_model`,
parametros LightGBM). `CLAUDE.md` seccion "Stack" ("LightGBM, NO
XGBoost: dataset es pequeno").

---

### P8. "¿Por que LangGraph y no un script simple con if/else?"

**Probabilidad:** MEDIA

**Respuesta:** La logica economica de decision SI es un script puro
(costs_v2.py, funciones sin estado). LangGraph es el wrapper que
agrega: (1) trazabilidad — cada nodo registra su estado para
auditoria, (2) explicabilidad — el LLM genera justificacion en
espanol de cada decision, (3) extensibilidad — agregar un nodo de
monitoreo o feedback sin tocar la logica economica. Si se quitara
LangGraph, las decisiones economicas serian identicas — el test de
equivalencia decisional lo demuestra.

**Soporte:** `src/agent/graph.py` linea 212 (StateGraph, topologia
en lineas 214–227). `tests/test_agent_decisions.py` (test de
equivalencia decisional con y sin LLM).
`docs/architecture.md` seccion de diseno del agente.

---

## Preguntas de negocio — Director Supply Chain

### P9. "¿Cuanto cuesta implementar esto en produccion?"

**Probabilidad:** ALTA

**Respuesta:** La infraestructura de nube cuesta entre $700 y $2,600
MXN al mes ($34–130 USD). La inversion real es la integracion de
datos: conectar los sistemas de inventarios y ventas de Herdez a la
nube, lo cual depende de la complejidad de sus sistemas actuales
(SAP, Oracle, etc.). Para el piloto de 2 meses en 1–2 centros, el
costo total de nube es menor a $6,000 MXN.

**Soporte:** `docs/architecture_gcp.md` seccion "Estimacion de
costos" (tabla de 8 servicios con rangos).

---

### P10. "¿En cuanto tiempo se paga solo?"

**Probabilidad:** ALTA

**Respuesta:** Depende de como operan hoy. Si no hay sistema de
priorizacion, el ahorro es de $2.6M MXN por periodo y el sistema se
paga en menos de un dia. Si ya priorizan transferencias manualmente,
el ahorro marginal es menor pero el sistema agrega automatizacion,
verificacion de seguridad del origen, y escalabilidad. La respuesta
honesta: el piloto de 2 meses mide el payback real con datos de
Herdez. No necesitamos proyectar — podemos medir.

**Soporte:** `docs/architecture_gcp.md` seccion "Contexto de
negocio: dos escenarios de ROI" (Escenarios A y B con tablas).

---

### P11. "¿Que pasa si el modelo falla en produccion?"

**Probabilidad:** ALTA

**Respuesta:** El sistema tiene fallback automatico: si el modelo se
degrada, la priorizacion cambia a la regla by_costo_quiebre, que de
hecho gana al modelo en el prototipo. Las decisiones economicas
siguen siendo validas porque dependen de costs_v2.py, no del modelo.
Ademas, el monitoreo de drift alerta automaticamente cuando las
predicciones del modelo se desvian de lo esperado, antes de que el
impacto sea visible en la operacion.

**Soporte:** `data/backtest_results.parquet` (by_costo_quiebre:
~$950k/fold, 2° lugar). `docs/architecture_gcp.md` seccion
"Validacion de calidad" (monitoreo de drift, CI que bloquea merges).

---

### P12. "¿Necesitamos contratar gente nueva para operar esto?"

**Probabilidad:** ALTA

**Respuesta:** No se requiere un equipo dedicado de ciencia de datos.
La integracion inicial la puede hacer el area de TI existente con
acompanamiento del proveedor (~2 meses). Para operacion continua,
se necesita ~0.5 persona de TI/analitica para monitoreo y
actualizaciones. El equipo de Supply Chain solo necesita capacitacion
en el uso del dashboard — no necesitan entender como funciona el
modelo por dentro.

**Soporte:** `docs/architecture_gcp.md` seccion "Estimacion de
costos" (nota sobre costo de personal). Slide 14 del outline
(seccion "Que necesita para operar").

---

### P13. "¿Esto funciona con nuestros 200+ SKUs, no solo los 5 del prototipo?"

**Probabilidad:** ALTA

**Respuesta:** La arquitectura esta disenada para eso. BigQuery
maneja millones de filas sin cambio de codigo. Vertex AI Pipelines
reentrena con cualquier volumen. Cloud Run escala automaticamente.
Lo que cambia con 200+ SKUs es la riqueza del modelo: mas productos
significa mas patrones para aprender, y ahi es donde el modelo
supera a las reglas fijas. El piloto valida con un subconjunto real
(10–20 SKUs de una categoria) antes de expandir.

**Soporte:** `docs/architecture_gcp.md` seccion "Mapeo prototipo →
produccion" (columna "Que cambia" para cada componente).

---

### P14. "¿Que garantias tenemos de que funcione?"

**Probabilidad:** ALTA

**Respuesta:** Ninguna garantia sin piloto — y desconfiaria de
quien ofreciera una. Lo que tenemos es evidencia de prototipo: la
logica de decision funciona, el ahorro simulado es de 71%, y la
arquitectura es solida. La garantia real viene del piloto: 2 meses
en 1–2 centros, corriendo en paralelo con la operacion actual,
midiendo si el sistema hubiera tomado mejores decisiones. Si no
demuestra valor medible, no se expande.

**Soporte:** `data/backtest_results.parquet` (evidencia de
simulacion). Slide 17 del outline (criterios de Go/No-Go).

---

### P15. "¿Por que no simplemente contratar mas gente de almacen?"

**Probabilidad:** MEDIA-ALTA

**Respuesta:** El problema no es falta de manos en el almacen — es
falta de informacion para decidir que mover y cuando. Con 200
productos y 15 centros, nadie puede calcular mentalmente cuales son
las 3 transferencias mas rentables del dia. El sistema no reemplaza
al equipo; le da la informacion que necesita para tomar mejores
decisiones con la misma gente.

**Soporte:** `docs/architecture.md` lineas 5–20 (restriccion de
N=3: el cuello de botella es coordinacion, no mano de obra).

---

### P16. "¿Se puede hacer un piloto antes de invertir fuerte?"

**Probabilidad:** ALTA

**Respuesta:** Eso es exactamente lo que proponemos. Piloto de 2
meses en 1–2 centros de distribucion, costo de nube menor a $6,000
MXN total, corriendo en paralelo sin afectar la operacion actual.
Al final del piloto hay datos concretos para decidir si expandir
o no. Los criterios de exito estan definidos antes de empezar, no
despues.

**Soporte:** Slide 17 del outline (criterios de Go/No-Go).
`docs/architecture_gcp.md` seccion "Estimacion de costos"
($34–130 USD/mes × 2 meses).

---

## Preguntas trampa

### P17. "Si una regla simple gana, ¿para que necesitamos ML?"

**Probabilidad:** ALTA

**Respuesta:** La regla simple gana en priorizacion pura con 60
dias y 5 productos. Pero el sistema completo hace mas que priorizar:
verifica automaticamente que el centro que envia no quede en riesgo
(safety check), calcula el volumen exacto de unidades a mover,
genera explicacion auditable de cada decision, y escala a 200+
productos sin aumentar personal. Ademas, con datos reales la
regla fija pierde ventaja: en Cuaresma el atun sube de riesgo,
en Buen Fin la demanda cambia — la regla no se adapta, el modelo si.

**Soporte:** `src/economics/costs_v2.py` funcion `select_origin()`
(safety check del origen). `docs/architecture_gcp.md` seccion
"Contexto de negocio: dos escenarios de ROI" (Escenario B: valor
mas alla de la priorizacion).

---

### P18. "¿Como sabemos que los $2.6M MXN de ahorro son reales y no solo una proyeccion optimista?"

**Probabilidad:** ALTA

**Respuesta:** Son resultado de una simulacion con datos sinteticos,
no una proyeccion sobre operacion real. El numero dice: "si
hubieramos tenido el sistema corriendo durante estos 60 dias, el
costo habria sido $1.05M en vez de $3.6M". Para saber cuanto ahorra
en la realidad de Herdez, necesitamos el piloto. Por eso no estamos
pidiendo aprobacion del sistema completo — estamos pidiendo
aprobacion de 2 meses de medicion.

**Soporte:** `data/backtest_results.parquet` (simulacion, no
proyeccion). Slide 17 del outline (piloto como mecanismo de
validacion).

---

## Resumen de probabilidades

| Prob. | Preguntas |
|-------|-----------|
| **ALTA** | P1, P2, P3, P5, P9, P10, P11, P12, P13, P14, P16, P17, P18 |
| **MEDIA-ALTA** | P4, P15 |
| **MEDIA** | P6, P7, P8 |

**Total: 18 preguntas** (8 tecnicas + 8 negocio + 2 trampa).

Las 13 de probabilidad ALTA cubren el 90% de lo que probablemente
pregunten. Las respuestas estan disenadas para reconocer limitaciones
honestamente y redirigir hacia el piloto como mecanismo de validacion.
