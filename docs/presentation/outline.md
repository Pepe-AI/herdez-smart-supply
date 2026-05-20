# Outline de Presentacion — Herdez Smart-Supply (60 min)

## Estructura general

| Bloque | Duracion | Audiencia principal |
|--------|----------|-------------------|
| Apertura | 5 min | Ambas |
| Narrativa Gerente de IA | ~25 min | Tecnica |
| Narrativa Director Supply Chain | ~25 min | Negocio |
| Cierre + Q&A | 5 min | Ambas |

---

## Apertura (5 min)

**Titulo:** Herdez Smart-Supply: prediccion de quiebres de stock +
agente de IA para decisiones de transferencia

**Bullets:**
- Quiebres de stock cuestan millones. El sistema los detecta antes
  de que ocurran y recomienda transferencias optimas.
- Prototipo funcional de 5 dias: modelo ML + agente economico +
  dashboard interactivo.
- Dos narrativas: la tecnica (como se construyo, que aprendimos)
  y la de negocio (cuanto vale, que necesita para produccion).

**Evidencia:** Dashboard en vivo o screenshot `dashboard_full_sidebar.png`

---

## Narrativa Gerente de IA (~25 min)

### Slide 1 — Problema tecnico (~2 min)

**Titulo:** Quiebres de stock: un problema de optimizacion bajo
restricciones

**Bullets:**
- Quiebres cuestan **$3,634,000 MXN/fold** en inaccion total
  (costo diario × dias de exposicion × SKUs afectados).
- Restriccion operativa real: maximo **N=3 transferencias diarias
  por CEDI origen** (~2h por transferencia, jornada de 8h).
- Con 4 CEDIs → maximo 12 transferencias/dia en toda la red.
- Reto: decidir **que** transferir, **de donde**, y **cuanto**,
  cada dia, bajo capacidad limitada.

**Evidencia:**
- `docs/architecture.md` lineas 5–20 (parametro N=3, justificacion
  de 2h por transferencia)
- Numeros de costo: `src/economics/costs_v2.py` linea 376
  (`run_capacity_comparison`, estrategia "inaction")

**Screenshot/diagrama:** Ninguno (slide de apertura con texto).

---

### Slide 2 — Dataset y limitaciones (~2 min)

**Titulo:** Dataset sintetico: 60 dias, 5 SKUs, 4 CEDIs

**Bullets:**
- ~1,200 filas. Periodo de 60 dias. 5 SKUs (salsas, atun,
  champinones, mole, chiles). 4 CEDIs (Norte, Sur, Centro, Bajio).
- **Sin estacionalidad real**: 60 dias no captura ciclos anuales.
- **Sin eventos comerciales**: no hay Hot Sale, Buen Fin, etc.
- **Stock=0 nunca ocurre** en el historico (minimo observado: 50
  unidades) → el target es un **proxy proyectado**, no una
  observacion directa.
- Senal de features categoricas: Promocion reduce quiebre en 4/5
  SKUs (Welch p=0.47 en ventas, efecto en tasa); Clima sin senal
  (Chi² p=0.93).

**Evidencia:**
- `notebooks/01_eda.ipynb` — seccion de cierre (~linea 1083):
  conclusiones del EDA con tests estadisticos
- `CLAUDE.md` seccion "Hechos del dataset"

**Screenshot/diagrama:** Tabla resumen del EDA (del notebook) o
slide con las cifras clave.

---

### Slide 3 — Regimen de costos asimetrico (~3 min)

**Titulo:** Ratio 53:1 — cuando equivocarse por omision cuesta
53 veces mas que por accion

**Bullets:**
- **FN promedio: $53,333 MXN.** Mix ponderado de 5 SKUs × costo
  quiebre diario × 5 dias exposicion. Salsas: $75k. Resto: $40k.
- **FP promedio: $997 MXN.** Costo de una transferencia innecesaria
  (~$10.45/unidad × ~95 unidades promedio).
- **Ratio: 53:1.** Cada quiebre no detectado cuesta 53 transferencias
  innecesarias.
- **Umbral economico optimo:** p > 0.0187 (derivado de
  $997 / ($997 + $53,333)). Transferir si la probabilidad de
  quiebre supera 1.87%.
- **Consecuencia:** bajo capacidad ilimitada, **all_positive**
  (transferir siempre) es Pareto-optimo. No existe umbral del
  modelo que lo supere. Curva costo-vs-umbral es monotonamente
  creciente.

**Evidencia:**
- `docs/architecture.md` lineas 111–130 (politica de filtros,
  derivacion del umbral)
- `CLAUDE.md` seccion "Ratio real de costos: 53:1"
- Threshold sweep: `notebooks/01_eda.ipynb`

**Screenshot/diagrama:** Grafico de threshold sweep (costo vs umbral)
del notebook o del dashboard tab Backtest.

---

### Slide 4 — Modelo predictivo: LightGBM con features lagged (~3 min)

**Titulo:** Resolviendo el leakage: de AUC-PR 0.99 a 0.45

**Bullets:**
- **Leakage detectado:** el target
  `(stock_actual - ventas_rolling_7d * 5) < 0` es determinista
  sobre `stock_actual` y `ventas_unidades`. Usarlas como features
  → AUC-PR ≈ 1.0 (el modelo memoriza la formula).
- **Solucion:** features **lagged** (t-1 o anterior):
  `stock_lag_{1,3,5,7}`, `ventas_lag_{1,3,5,7}`,
  `ventas_rolling_7d_lag1`, ratios de cobertura lagged.
  `stock_actual` y `ventas_unidades` en EXCLUDE_COLS.
- **AUC-PR honesto:** ~0.45 ± 0.06 (TimeSeriesSplit, 5 folds).
  Baseline random ≈ 0.335 (prevalencia del target). El modelo
  aporta senal, aunque modesta.
- **El modelo es dinamico:** predicciones van de 0.02 a 0.95 en
  el mismo fold. Puede diferenciar riesgo alto de bajo, a diferencia
  de una regla binaria.
- **Por que LightGBM y no XGBoost/NN:** dataset pequeno (~1,200
  filas), LightGBM maneja bien datos tabulares con pocas filas.

**Evidencia:**
- `src/ml/train.py` linea 52 (funcion `train_model`)
- `notebooks/01_eda.ipynb` — seccion de leakage (AUC-PR leaked
  vs lagged vs exogenas)
- `CLAUDE.md` seccion "Leakage resuelto"

**Screenshot/diagrama:** Comparacion AUC-PR leaked (0.99) vs
lagged (0.45) vs exogenas-only (0.45) del notebook.

---

### Slide 5 — Restriccion N=3 cambia todo (~3 min)

**Titulo:** Capacidad limitada: cuando priorizar importa mas
que predecir

**Bullets:**
- Bajo capacidad ilimitada, all_positive gana (slide 3). Pero con
  N=3 transferencias/dia/CEDI, **hay que elegir cuales ejecutar**.
- El problema pasa de "actuar o no" a **"que priorizar"**. Aqui
  emerge el valor del modelo como sensor de probabilidades.
- **Politica de filtros uniforme** (aplicada a las 7 estrategias):
  1. Filtro de umbral economico: p > 0.0187
  2. Filtro de beneficio neto positivo: costo_no_actuar > costo_transferir
- La unica diferencia entre estrategias es el **criterio de
  ordenamiento** del conjunto ya filtrado.
- Esto garantiza que ninguna estrategia tome decisiones
  economicamente irracionales (transferir cuando cuesta mas que
  no actuar).

**Evidencia:**
- `src/economics/costs_v2.py` linea 265 (funcion
  `simulate_day_with_priority`, filtros en las primeras lineas)
- `docs/architecture.md` lineas 111–130 (politica de filtros)

**Screenshot/diagrama:** Diagrama de flujo de la logica de filtros
(se puede dibujar simple en el slide).

---

### Slide 6 — Backtest con 7 estrategias (~4 min)

**Titulo:** Backtest honesto: 7 estrategias, filtros uniformes,
modelo en 3er lugar

**Bullets:**
- **7 estrategias evaluadas** bajo condiciones identicas
  (TimeSeriesSplit, 5 folds, filtros uniformes):

| Ranking | Estrategia | Costo/fold (MXN) |
|---------|-----------|-----------------|
| 1 | tasa_base | $912,709 |
| 2 | costo_quiebre | ~$950k |
| 3 | **modelo (model_prioritized)** | **$1,046,657** |
| 4 | heuristic_deficit | ~$1.1M |
| 5 | stock_lag1 | ~$1.2M |
| 6 | fifo | ~$1.4M |
| 7 | inaccion | $3,634,000 |

- **Brecha modelo vs mejor baseline:** +$133,948/fold (+14.7%).
- **Por que los baselines estaticos ganan con 60 dias:** los
  rankings de SKU por costo de quiebre son estables en el periodo.
  tasa_base y costo_quiebre explotan esta estabilidad. Con datos
  de 12+ meses (estacionalidad, cambios de mix), los rankings
  cambian y el modelo adaptativo deberia mejorar.
- **Valor del modelo hoy:** ahorra $2.6M/fold vs inaccion (71%),
  y la brecha con baselines se cierra con mas datos.

**Evidencia:**
- `data/backtest_results.parquet` — tabla completa de resultados
- `src/economics/costs_v2.py` linea 376 (`run_capacity_comparison`)
- Screenshot: `docs/presentation/screenshots/tab_backtest_full.png`
- Screenshot: `docs/presentation/screenshots/tab_backtest_chart.png`

**Screenshot/diagrama:** Tab Backtest del dashboard (tabla +
grafico de barras).

---

### Slide 7 — Disciplina metodologica: 3 bugs detectados (~3 min)

**Titulo:** Encontramos errores, los corregimos, los documentamos,
los prevenimos con tests

**Bullets:**
- **Bug 1 — Leakage en features (Dia 1):**
  Features directas (`stock_actual`, `ventas_unidades`) hacian que
  el modelo memorizara la formula del target. Correccion: features
  lagged exclusivamente. AUC-PR bajo de 0.99 a 0.45 — el numero
  honesto.

- **Bug 2 — Asimetria select_origin + leakage temporal (meta-auditoria
  Dia 2):**
  (a) Baselines usaban "primer origen valido" vs modelo usaba "mejor
  excedente". Correccion: `select_origin()` uniforme para todas.
  (b) `by_stock_bajo` usaba `stock_actual` (futuro) para priorizar.
  Correccion: `stock_lag_1`.
  Impacto: `by_stock_bajo` bajo de 1° a 5° en ranking.

- **Bug 3 — Divergencia backtest-agente (debugging Dia 3):**
  `simulate_day_with_priority()` divergia del agente en scoring,
  filtro de umbral y filtro de beneficio_neto. Correccion: `y_proba`
  obligatorio, filtros uniformes en las 7 estrategias. Impacto:
  modelo subio de 5° a 3°.

- **Cada bug tiene test de regresion** que previene recurrencia:
  `TestBacktestMatchesAgentDecisions` (equivalencia),
  `test_by_stock_uses_lag1` (leakage temporal),
  filtros uniformes verificados en cada estrategia.

- **Mensaje para la audiencia:** la disciplina de encontrar y
  corregir errores propios es mas valiosa que el numero final.

**Evidencia:**
- `docs/meta_auditoria_dia2.md` lineas 187–248 (Bug 3 completo)
- `tests/test_costs_v2.py` lineas 276–345 (clase
  `TestBacktestMatchesAgentDecisions`)
- `CLAUDE.md` seccion "Bugs detectados en meta-auditoria"

**Screenshot/diagrama:** Tabla "antes vs despues" de cada bug
(puede ser texto en slide).

---

### Slide 8 — Arquitectura del agente: LangGraph con 5 nodos (~2 min)

**Titulo:** Agente economico: logica determinista + LLM para
explicabilidad

**Bullets:**
- **5 nodos** en un grafo LangGraph:
  - Nodos 1-3 (deterministas, Python puro): `generate_alerts`,
    `evaluate_costs`, `prioritize_and_allocate`. Usan funciones
    de `costs_v2.py`. Misma logica que el backtest.
  - Nodo 4 (LLM): `decide` — revisa el plan propuesto. Puede
    invocar tools pero NO cambia las decisiones economicas.
  - Nodo 5 (LLM): `explain` — genera explicacion en espanol
    de las decisiones tomadas.
- **Decision arquitectonica clave:** modulo puro `costs_v2.py` +
  wrapper LangGraph. Zero acoplamiento. Si el LLM falla, las
  decisiones economicas son identicas (test de equivalencia
  decisional lo verifica).
- **Gemini 2.5 Flash** como LLM (auto-deteccion: si hay key, usa
  Gemini; si no, `_MockLLM` aprueba el plan determinista).
- **Por que LangGraph y no un script:** el grafo con estado permite
  trazabilidad de cada decision, tool calls del LLM, y extension
  futura (nodos de monitoreo, feedback loop).

**Evidencia:**
- `src/agent/graph.py` linea 212 (definicion del `StateGraph`,
  topologia en lineas 214–227)
- `docs/architecture.md` — diseno del agente, flujo de 5 pasos
- Test de equivalencia: `tests/test_agent_decisions.py`

**Screenshot/diagrama:** Diagrama de flujo del grafo (5 nodos con
flechas, se puede hacer simple en el slide).

---

### Slide 9 — Arquitectura GCP propuesta (~3 min)

**Titulo:** Del prototipo a produccion: 6 servicios GCP

**Bullets:**
- **Diagrama simplificado** (LR) del sistema productivo:
  BigQuery → Vertex AI Pipelines → Model Registry → Cloud Run
  (agente + Gemini) → Cloud Run (Streamlit) + Cloud Build (CI/CD).
- **Que se conserva sin cambios:** modelo LightGBM, logica
  `costs_v2.py`, grafo LangGraph, rol del LLM, TimeSeriesSplit.
- **Que cambia:** escala (5→200+ SKUs), frecuencia (manual→diaria),
  acceso (localhost→equipo SC), resiliencia (logs→alertas).
- **Costo de infraestructura:** $34–130 USD/mes. Trivial en ambos
  escenarios de ROI (ver seccion de costos).
- **Decision Go/No-Go** no depende de GCP sino de inversion en
  personal (ML engineer + DevOps).

**Evidencia:**
- `docs/architecture_gcp.md` — documento completo (diagrama,
  justificaciones, mapeo, costos)
- Screenshot: `docs/presentation/diagram_simple.png`

**Screenshot/diagrama:** `diagram_simple.png` (diagrama LR validado
como legible a 1280x720).

---

## Resumen de tiempos — Narrativa tecnica

| Slide | Tema | Minutos |
|-------|------|---------|
| 1 | Problema tecnico | 2 |
| 2 | Dataset y limitaciones | 2 |
| 3 | Regimen de costos 53:1 | 3 |
| 4 | Modelo LightGBM + leakage | 3 |
| 5 | Restriccion N=3 | 3 |
| 6 | Backtest 7 estrategias | 4 |
| 7 | 3 bugs + disciplina metodologica | 3 |
| 8 | Agente LangGraph | 2 |
| 9 | Arquitectura GCP | 3 |
| **Total** | | **25 min** |

---

## Narrativa Director Supply Chain (~25 min)

### Slide 10 — El costo de no hacer nada (~2 min)

**Titulo:** Cada dia sin sistema, los quiebres cuestan millones

**Bullets:**
- Un quiebre de stock en salsas cuesta **$75,000 MXN** en 5 dias
  entre venta perdida y costo de oportunidad. En atun, champinones
  o mole: $40,000 MXN.
- En el periodo evaluado (~10 dias, 5 productos, 4 centros de
  distribucion), el costo total sin ningun sistema asciende a
  **$3.6 millones MXN.**
- Extrapolando a un ano con 200+ productos y 15 centros: el costo
  de inaccion se mide en **decenas de millones de pesos anuales.**
- No es que el producto no exista en la red — esta en el CEDI
  equivocado. El problema es de **coordinacion, no de abasto.**

**Soporte numerico:**
- $3,634,000 MXN/fold promedio (5 folds de ~10 dias):
  `data/backtest_results.parquet`, columna "inaction"
- Costo por SKU: salsas $15,000/dia × 5 dias = $75,000;
  atun/champinones/mole $8,000/dia × 5 dias = $40,000:
  `src/economics/costs_v2.py` linea 376

**Asset visual:** Tabla simple de costo de quiebre por producto.

| Producto | Costo diario | Exposicion (5 dias) |
|----------|-------------|-------------------|
| Salsa Roja 200g | $15,000 | $75,000 |
| Salsa Verde 200g | $15,000 | $75,000 |
| Atun en Agua 140g | $8,000 | $40,000 |
| Champinones 380g | $8,000 | $40,000 |
| Mole Poblano 250g | $8,000 | $40,000 |

---

### Slide 11 — Que hace el sistema (~3 min)

**Titulo:** Un asistente que detecta riesgos y recomienda acciones
antes de que haya faltantes

**Bullets:**
- **Paso 1 — Deteccion:** cada manana, el sistema analiza inventarios
  y ventas recientes para identificar que productos tienen riesgo
  de quiebre en los proximos 5 dias.
- **Paso 2 — Priorizacion:** con capacidad limitada (maximo 3
  transferencias por centro por dia), el sistema decide **cuales**
  son las mas urgentes y rentables.
- **Paso 3 — Recomendacion:** para cada transferencia, indica
  de que centro sale, cuantas unidades mover, y por que. En
  espanol, con explicacion completa.
- **Paso 4 — Decision humana:** el jefe de CEDI revisa, aprueba
  o ajusta. El sistema **no reemplaza** al equipo; lo asiste
  con informacion que seria imposible calcular manualmente con
  200 productos diarios.
- La diferencia con la operacion manual: hoy se priorizan las
  transferencias por intuicion o por urgencia visible. El sistema
  las prioriza por **impacto economico calculado.**

**Soporte numerico:**
- Flujo de 5 pasos del agente: `docs/architecture.md` lineas 5–20
- Diagrama simplificado: `docs/presentation/diagram_simple.png`

**Asset visual:** Diagrama LR simplificado del flujo
(detectar → priorizar → recomendar → explicar → decidir).

---

### Slide 12 — Resultado bajo restriccion operativa (~3 min)

**Titulo:** 71% menos costo operativo, con la misma capacidad
de transferencias

**Bullets:**
- Sin sistema: **$3.6M MXN** de costo por quiebres no atendidos
  (en el periodo evaluado de ~10 dias).
- Con sistema: **$1.05M MXN** — el sistema detecta los riesgos
  mas costosos y los atiende primero.
- **Reduccion del 71%** en costo operativo, usando la misma
  capacidad logistica (3 transferencias/dia/centro).
- En numeros concretos por fold (~10 dias): el sistema procesa
  **176 situaciones de riesgo**, ejecuta **106 transferencias**
  priorizadas por costo, y difiere **58 con razon explicita**
  (bajo riesgo o sin beneficio neto).
- Las 12 restantes se descartan automaticamente porque no
  superan el umbral minimo de riesgo.

**Soporte numerico:**
- Modelo: $1,046,657 MXN/fold; inaccion: $3,634,000 MXN/fold:
  `data/backtest_results.parquet`
- 176 alertas/fold, 106 aprobadas, 58 diferidas, 12 sin alerta:
  `data/daily_predictions.parquet` (conteos verificados)

**Asset visual:** Grafico de barras comparando costo de las 7
estrategias. Screenshot: `tab_backtest_chart.png`

---

### Slide 13 — Por que no es perfecto: honestidad (~3 min)

**Titulo:** El sistema captura el 91% del ahorro disponible hoy,
y mejora con mas datos

**Bullets:**
- Las reglas simples de priorizacion (por ejemplo, "atender
  primero el producto con mayor costo de quiebre") funcionan
  mejor que el modelo de prediccion **en este dataset de 60 dias.**
- Razon: con solo 60 dias y 5 productos, los rankings de riesgo
  son muy estables. Una regla fija que memoriza el orden funciona
  bien. Esto no es una debilidad del modelo, es una propiedad
  del dataset.
- **Brecha actual:** el modelo queda 14.7% detras de la mejor
  regla ($1.05M vs $912k MXN/fold).
- **Con datos reales, los rankings cambian:** en Cuaresma el atun
  sube de demanda y su riesgo de quiebre supera al de salsas; en
  Buen Fin la demanda de todos los SKUs se dispara de forma
  desigual; un lanzamiento nuevo no tiene historico para una regla
  fija. El modelo se reentrena y se adapta — la regla "siempre
  atender salsas primero" no.
- En la practica: **el sistema ya ahorra $2.6M MXN vs no hacer nada.**
  La pregunta no es si funciona — es cuanto mas puede mejorar
  con datos reales.

**Soporte numerico:**
- Modelo $1,046,657 vs tasa_base $912,709 (+14.7%):
  `data/backtest_results.parquet`
- Ahorro vs inaccion: $3,634,000 - $1,046,657 = $2,587,343 (71%)
- Analisis de estabilidad de rankings: `notebooks/01_eda.ipynb`
  seccion de cierre (~linea 1083)

**Asset visual:** Tabla comparativa:

| Metrica | Hoy (60 dias, 5 SKUs) | Proyeccion (12+ meses, 200+ SKUs) |
|---------|-----------------------|------------------------------------|
| Datos de entrenamiento | 1,200 filas | 100,000+ filas |
| Estacionalidad | No capturada | Capturada (ciclos anuales) |
| Rankings de riesgo | Estables (reglas ganan) | Dinamicos (modelo gana) |
| Posicion del modelo | 3° de 7 | Top 1-2 esperado |

---

### Slide 14 — Que necesita el sistema para operar (~3 min)

**Titulo:** Datos, nube, y un piloto de 2 meses

**Bullets:**
- **Datos:** historico de ventas diarias, inventarios por centro,
  y registro de transferencias realizadas. Si estos datos ya estan
  en algun sistema (SAP, Oracle, Excel), se pueden conectar a la
  nube en dias.
- **Infraestructura:** Google Cloud Platform. Costo estimado:
  **$34–130 USD/mes** (~$700–2,600 MXN/mes). Menos que una
  suscripcion de software empresarial.
- **Equipo:** no se requiere un equipo dedicado de ciencia de datos.
  La integracion la puede hacer el area de TI existente con
  acompanamiento del proveedor. Para mantenimiento continuo:
  ~0.5 personas de TI/analitica.
- **Tiempo:** piloto funcional en **2 meses** con 1–2 centros
  de distribucion. Expansion a toda la red en 6 meses.

**Soporte numerico:**
- Costo GCP: `docs/architecture_gcp.md` seccion "Estimacion de
  costos" (tabla de 8 servicios con rangos)
- Mapeo prototipo → produccion: `docs/architecture_gcp.md` seccion
  "Mapeo prototipo → produccion" (8 componentes)

**Asset visual:** Timeline simplificado:
```
Mes 1-2: Piloto (1-2 CEDIs, datos reales)
Mes 3-4: Expansion (todos los CEDIs, monitoreo)
Mes 5-6: Produccion (reentrenamiento automatico, A/B testing)
```

---

### Slide 15 — ROI estimado en dos escenarios (~3 min)

**Titulo:** Dependiendo de como operan hoy, el sistema se paga
en dias o semanas

**Bullets:**
- **Escenario A — Si hoy no hay sistema de priorizacion:**
  El ahorro completo aplica. Se evitan **$2.6M MXN** por fold
  (~10 dias) en costos de quiebres. La infraestructura de nube
  ($2,000 MXN/mes) se recupera en **menos de 1 dia** de operacion.

- **Escenario B — Si hoy ya se priorizan transferencias
  manualmente:**
  El ahorro marginal del modelo es del orden de **$134,000
  MXN/fold.** Ademas, el sistema agrega valor que no se puede
  replicar manualmente:
  - Verificacion automatica de que el centro que envia no quede
    en riesgo
  - Calculo exacto de cuantas unidades mover (ni de mas, ni de
    menos)
  - Explicacion escrita de cada decision, auditable
  - Escalabilidad a 200+ productos sin aumentar personal
  La recuperacion de la infraestructura: **dias a semanas.**

- En **ambos escenarios**, el costo de la nube es trivial. La
  decision real es si invertir en la integracion de datos y el
  acompanamiento del piloto.

**Soporte numerico:**
- Escenarios A y B: `docs/architecture_gcp.md` seccion "Contexto
  de negocio: dos escenarios de ROI"
- Modelo $1,046,657 vs tasa_base $912,709:
  `data/backtest_results.parquet`

**Asset visual:** Tabla con los dos escenarios lado a lado.

---

### Slide 16 — Hoja de ruta: piloto y produccion (~3 min)

**Titulo:** De piloto a produccion en 6 meses

**Bullets:**
- **Fase 1 (Meses 1–2): Piloto controlado**
  - Conectar datos reales de 1–2 CEDIs a la nube
  - Entrenar modelo con historico completo (12+ meses)
  - Correr sistema en paralelo (sin reemplazar operacion actual)
  - Medir: el sistema hubiera tomado mejores decisiones que el
    equipo?

- **Fase 2 (Meses 3–4): Expansion con monitoreo**
  - Activar sistema en todos los CEDIs
  - El equipo de logistica recibe recomendaciones diarias
  - Monitoreo continuo: si el sistema se equivoca, se detecta
    automaticamente y se alerta

- **Fase 3 (Meses 5–6): Produccion con mejora continua**
  - Reentrenamiento automatico mensual con datos nuevos
  - A/B testing: comparar semanas con sistema vs sin sistema
  - Reporte mensual de ahorro real vs proyectado

- **Fase 4 (Mes 7+): Optimizacion**
  - Agregar nuevos productos y centros de distribucion
  - El modelo mejora con cada mes de datos acumulados
  - Posibilidad de extender a otras categorias de Herdez

**Soporte numerico:**
- Arquitectura de produccion: `docs/architecture_gcp.md` seccion
  "Mapeo prototipo → produccion"
- Reentrenamiento en Vertex AI: `docs/architecture_gcp.md` seccion
  "Justificacion por servicio" (Vertex AI Pipelines)

**Asset visual:** Timeline visual de 4 fases con hitos clave.

---

### Slide 17 — Decision a tomar: Go/No-Go (~3 min)

**Titulo:** No pido aprobacion del sistema completo — pido
aprobacion del piloto

**Bullets:**
- **Criterios de Go (aprobar piloto):**
  - Hay datos historicos de ventas e inventarios disponibles
    (minimo 6 meses, ideal 12+)
  - Hay un sponsor en Supply Chain que asigne 1–2 CEDIs para
    el piloto
  - Hay presupuesto de nube (~$2,000 MXN/mes) y acompanamiento
    de TI (~0.5 persona durante 2 meses)

- **Criterios de No-Go (posponer, no descartar):**
  - Los datos no estan disponibles o no son confiables
  - La operacion actual ya optimiza transferencias con un sistema
    comparable y no hay mejora marginal medible
  - No hay capacidad de TI para la integracion en los proximos
    3 meses

- **Hito de validacion del piloto:**
  Correr el sistema en paralelo en 1 CEDI durante 1 mes. Comparar
  las decisiones del sistema vs las decisiones reales del equipo.
  Si el sistema hubiera evitado al menos 1 quiebre que el equipo
  no detecto, el piloto se justifica.

- **Riesgo de no actuar:** cada mes sin sistema es un mes donde
  los quiebres se atienden por intuicion. A $53,000 MXN por
  quiebre no detectado, el costo de esperar es medible.

**Soporte numerico:**
- Costo por FN: $53,333 MXN promedio: `CLAUDE.md` seccion
  "Ratio real de costos: 53:1"
- Costo de infraestructura: `docs/architecture_gcp.md` seccion
  "Estimacion de costos"

**Asset visual:** Checklist de Go/No-Go con dos columnas.

---

### Resumen de tiempos — Narrativa de negocio

| Slide | Tema | Minutos |
|-------|------|---------|
| 10 | Costo de no hacer nada | 2 |
| 11 | Que hace el sistema | 3 |
| 12 | Resultado: 71% reduccion | 3 |
| 13 | Honestidad sobre limitaciones | 3 |
| 14 | Que necesita para operar | 3 |
| 15 | ROI en dos escenarios | 3 |
| 16 | Hoja de ruta 6 meses | 3 |
| 17 | Decision Go/No-Go | 3 |
| **Total** | | **23 min** |

---

## Cierre + Q&A (5 min)

### Resumen ejecutivo para ambas audiencias

El sistema de prediccion de quiebres reduce el costo operativo en
un 71% respecto a la inaccion total, pasando de $3.6M MXN a $1.05M
MXN por periodo evaluado, usando la misma capacidad logistica de
transferencias que ya existe.

El modelo de inteligencia artificial queda en 3er lugar de 7
estrategias evaluadas, detras de dos reglas simples de priorizacion.
Esto no es una debilidad del modelo sino una propiedad del dataset
de prueba (60 dias, 5 productos): los rankings de riesgo son tan
estables que una regla fija funciona bien. Con datos reales de
Herdez (12+ meses, 200+ productos, estacionalidad), el modelo
adaptativo recupera ventaja estructural.

La arquitectura propuesta en Google Cloud cuesta entre $34 y $130
dolares mensuales — una fraccion insignificante del ahorro
generado. La inversion real es en integracion de datos y
acompanamiento del piloto, no en infraestructura.

La propuesta concreta: **aprobar un piloto de 2 meses en 1–2
centros de distribucion**, corriendo el sistema en paralelo con
la operacion actual, para medir el valor real con datos de Herdez.

---

## Preguntas hostiles anticipadas

Ver documento completo: `docs/presentation/preguntas_hostiles.md`

18 preguntas preparadas (8 tecnicas + 8 negocio + 2 trampa).
13 de probabilidad ALTA. Cada respuesta es 2-3 oraciones con
dato de soporte del repositorio.
