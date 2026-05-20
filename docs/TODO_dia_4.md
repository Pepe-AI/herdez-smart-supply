# TODO Dia 4 — Arquitectura GCP + Esqueleto de Presentacion

## Estado al inicio del Dia 4

Sistema implementado: pipeline ML, costs_v2.py con filtros uniformes,
agente LangGraph con Gemini opcional, dashboard Streamlit 3 tabs,
48 tests, 3 bugs documentados con tests de regresion.

Numeros headline (produccion, fuente de verdad):
- Modelo: $1,046,657/fold (3ro de 7)
- Mejor baseline (tasa_base): $912,709/fold
- Brecha: +$133,948/fold (+14.7%)
- Ahorro vs inaccion: $2,587,343/fold (71%)

---

## Bloque 1 — Diagrama y estructura de arquitectura GCP (~1.5h)

**Que:** Crear `docs/architecture_gcp.md` con el diagrama de servicios
GCP en Mermaid y la estructura de secciones (sin contenido detallado).

**Tareas:**
1. Diagrama Mermaid con los 5 servicios core: BigQuery (datos),
   Vertex AI Pipelines (reentrenamiento), Cloud Run (agente + API),
   Gemini API (explicaciones), Cloud Build + GitHub Actions (CI/CD).
   El diagrama debe incluir:
   - Flechas de flujo de datos explicitas entre servicios
   - El "happy path" de una alerta: desde su deteccion (datos en
     BigQuery → prediccion en Vertex AI → alerta al agente en
     Cloud Run → decision economica → explicacion via Gemini →
     visualizacion en dashboard)
2. Secciones vacias con encabezados: Diagrama, Justificacion por
   servicio, Mapeo prototipo-produccion, Estimacion de costos.
3. Conexion DuckDB → BigQuery como decision ya tomada (CLAUDE.md
   menciona "sintaxis compatible con BigQuery").

**Criterio de completada:** El archivo existe con diagrama renderizable
que muestra flujo de datos end-to-end y secciones con encabezados.
No necesita contenido en las secciones.

**Dependencias:** Ninguna.

**Tiempo estimado:** 1.5h.

---

## Bloque 2 — Justificacion pieza por pieza del stack GCP (~1.5h)

**Que:** Llenar la seccion de justificacion en architecture_gcp.md.

**Tareas:**
1. Para cada servicio: que resuelve, por que se eligio, que
   alternativa se descarto y por que.
2. Conectar con hallazgos del prototipo:
   - Regimen 53:1 → por que el modelo necesita reentrenamiento
     continuo (Vertex AI Pipelines)
   - Dataset de 60 dias → por que con 12+ meses mejora
     (BigQuery como data warehouse escalable)
   - N=3 transferencias → por que el agente corre diariamente
     (Cloud Run como servicio stateless)
   - Explicaciones en espanol → por que Gemini esta integrado
     (Gemini API con fallback)
3. Tabla: servicio | problema que resuelve | alternativa descartada |
   razon de descarte.

**Criterio:** Cada servicio tiene justificacion con referencia al
prototipo. La tabla esta completa.

**Dependencias:** Bloque 1 (estructura del documento).

**Tiempo estimado:** 1.5h.

---

## Bloque 3 — Mapeo prototipo → produccion (~1h)

**Que:** Tabla que muestra que cambia entre prototipo y produccion
para cada componente del sistema.

**Tareas:**
1. Tabla con columnas: Componente | Prototipo (hoy) | Produccion
   (propuesta) | Que cambia | Justificacion.
2. Componentes: Storage (DuckDB → BigQuery), ML (LightGBM local →
   Vertex AI Pipeline), Agente (LangGraph local → Cloud Run),
   LLM (Gemini API gratuita → Gemini API Enterprise), Dashboard
   (Streamlit local → Streamlit en Cloud Run o App Engine),
   CI/CD (manual → Cloud Build + GitHub Actions), Monitoreo
   (logs locales → Cloud Logging + alertas), Validacion de
   calidad (3 tests regresion locales → CI/CD que bloquea
   merges + monitoreo de drift).
3. Estimacion cualitativa de costos: ordenes de magnitud
   ($X/mes para cada servicio), no numeros exactos.

**Criterio:** Tabla completa con los 8 componentes (incluyendo
validacion de calidad). Estimacion de costos con rangos razonables.

**Dependencias:** Bloque 2 (justificaciones informan la tabla).

**Tiempo estimado:** 1h.

---

## Bloque 4 — Esqueleto de presentacion: narrativa Gerente IA (~1.5h)

**Que:** Crear `docs/presentation/outline.md` con la estructura
de 60 minutos y la narrativa tecnica (~30 min).

**Tareas:**
1. Crear directorio `docs/presentation/`.
2. Capturar screenshots del dashboard (3 tabs) para que los slides
   tecnicos puedan referenciarlos visualmente. Guardar en
   `docs/presentation/screenshots/`.
3. Estructura general: apertura (5 min), narrativa tecnica (25 min),
   narrativa negocio (25 min), cierre + preguntas (5 min).
4. Narrativa Gerente IA — slides con evidencia. Orden: problema →
   costos → modelo → restriccion → backtest → bugs → agente → GCP.
   - Slide 1: Problema (quiebres cuestan $3.6M/fold)
   - Slide 2: Dataset y limitaciones (60 dias, 5 SKUs, 4 CEDIs)
   - Slide 3: Regimen de costos 53:1 y umbral 0.0187
   - Slide 4: Leakage y solucion (AUC-PR leaked=0.99 vs lagged=0.45)
   - Slide 5: all_positive Pareto bajo capacidad ilimitada
   - Slide 6: Restriccion N=3 y por que cambia todo
   - Slide 7: Backtest 7 estrategias (tabla + grafico + screenshot)
   - Slide 8: Arquitectura del agente LangGraph (5 nodos)
   - Slide 9: 3 bugs encontrados y corregidos (cierre de disciplina
     metodologica: "encontramos errores, los corregimos, los
     documentamos, los prevenimos con tests")
   - Slide 10: Arquitectura GCP propuesta (diagrama)
   - Slide 11: Demo en vivo del dashboard (si aplica)
5. Cada slide con: titulo, 3-5 bullets, evidencia del repo (archivo
   y linea), tiempo estimado.

**Criterio:** Outline con 11+ slides, cada uno con evidencia mapeada.
Screenshots capturados para slides que los necesiten.

**Dependencias:** Bloques 1-3 (necesita el documento GCP para slide 10).

**Tiempo estimado:** 1.5h.

---

## Bloque 5 — Esqueleto de presentacion: narrativa Director SC (~1.5h)

**Que:** Agregar la narrativa de negocio (~30 min) al outline.

**Tareas:**
1. Narrativa Director Supply Chain — slides con ROI:
   - Slide 12: Costo de no hacer nada ($3.6M/fold = $X anualizados)
   - Slide 13: Que hace el sistema (explicacion sin jerga tecnica)
   - Slide 14: Resultado: ahorro de $2.6M/fold vs inaccion (71%)
   - Slide 15: Por que no es perfecto y que falta (honestidad)
   - Slide 16: Que necesita el sistema para funcionar (datos, infra)
   - Slide 17: Hoja de ruta 6 meses (quick wins + largo plazo)
   - Slide 18: Inversion estimada (GCP + equipo) vs ahorro esperado
   - Slide 19: Decision a tomar (Go/No-Go con criterios claros)
2. Cada slide con: titulo, 3-5 bullets en lenguaje de negocio,
   numero de soporte del repo.
3. Seccion de cierre: resumen ejecutivo de 1 parrafo para ambas
   audiencias.

**Criterio:** Outline con 8+ slides de negocio, lenguaje accesible.

**Dependencias:** Bloque 4 (estructura general ya definida).

**Tiempo estimado:** 1.5h.

---

## Bloque 6 — Anticipacion de preguntas hostiles (~1h)

**Que:** Lista de 10-15 preguntas probables con respuesta preparada.

**Tareas:**
1. Preguntas del Gerente de IA (tecnicas):
   - "¿Por que el modelo pierde 14.7% vs una regla simple?"
   - "¿AUC-PR de 0.45 es aceptable?"
   - "¿Por que LightGBM y no XGBoost/NN?"
   - "¿60 dias son suficientes para entrenar?"
   - "¿Como se valida que no hay leakage?"
   - "¿Que pasa si el regimen de costos cambia?"
   - "¿Por que LangGraph y no un script simple?"
2. Preguntas del Director SC (negocio):
   - "¿Cuanto cuesta implementar esto en produccion?"
   - "¿En cuanto tiempo se paga solo?"
   - "¿Que pasa si el modelo falla?"
   - "¿Necesitamos contratar gente nueva?"
   - "¿Esto funciona con nuestros otros 200 SKUs?"
   - "¿Por que no simplemente contratar mas gente de almacen?"
   - "¿Que garantias tenemos de que funcione?"
   - "¿Se puede hacer un piloto antes de invertir?"
3. Cada respuesta: 2-3 oraciones con dato de soporte del repo.

**Criterio:** 14+ preguntas con respuesta, organizadas por audiencia.

**Dependencias:** Bloques 4-5 (las preguntas derivan de la narrativa).

**Tiempo estimado:** 1h.

---

## Bloque 7 — Cierre del Dia 4 (~30min)

**Tareas:**
1. Actualizar CLAUDE.md:
   - Seccion Dia 4 con hallazgos
   - Corregir "46 total" → "48 total" en seccion Comandos
2. Actualizar meta_auditoria_dia2.md:
   - Lineas 219-222: corregir que y_proba ya no es opcional
   - Lineas 238-248: actualizar tabla con numeros de produccion
     finales (filtros uniformes)
3. Verificar que evidencia_ml_herdez_v3.docx (ya actualizado con
   numeros $1,046,657, 14.7%, 71% reduccion, Bug 3) esta disponible
   como anexo entregable en el repo o como archivo separado.
4. Generar `docs/TODO_dia_5.md`:
   - Pulir slides (formato final)
   - Ensayo de la presentacion
   - Preparacion del demo en vivo
   - Mejoras de implementacion detectadas durante Dia 4 (si las hay)
5. Commits atomicos + push.

**Criterio:** CLAUDE.md actualizado, TODO_dia_5 generado, push limpio.

**Dependencias:** Bloques 1-6 completos.

**Tiempo estimado:** 30min.

---

## Riesgos identificados

### R1: Formato de la presentacion
**Riesgo:** ¿Slides (PowerPoint/Google Slides) o markdown renderizado?
**Impacto:** Afecta si el esqueleto es un outline.md o un .pptx.
**Mitigacion:** Generar outline.md como fuente de verdad; el usuario
decide el formato final. El outline tiene suficiente detalle para
convertir a cualquier formato.
**Accion requerida:** Confirmar con el usuario antes del Bloque 4.

### R2: Diagrama GCP existente
**Riesgo:** ¿Hay diagrama del brief original (Prueba_Tecnica_Herde_IA.pdf)?
**Impacto:** Si existe, se reutiliza; si no, se crea desde cero.
**Mitigacion:** No pude leer el PDF. Crear diagrama desde cero con
Mermaid; si el PDF tiene uno, se adapta despues.

### R3: Demo en vivo del dashboard
**Riesgo:** ¿Se hara demo en vivo durante la presentacion?
**Impacto:** Si si, hay que validar end-to-end con datos frescos.
**Dependencias del demo:**
  - Parquets precomputados (data/backtest_results.parquet,
    data/daily_predictions.parquet)
  - DuckDB con datos (data/herdez.db)
  - Python + uv + dependencias instaladas
  - GOOGLE_API_KEY para chat con Gemini (opcional, hay fallback mock)
**Mitigacion:** El dashboard funciona sin API key (modo demo). Los
parquets estan commiteados. Solo falta tener el entorno instalado.

### R4: API key de Gemini en presentacion
**Riesgo:** Rate limits de la API gratuita durante demo en vivo.
**Impacto:** El chat podria fallar durante la presentacion.
**Mitigacion:** El fallback mock funciona automaticamente. Preparar
screenshots del chat con Gemini real como backup. La tier gratuita
de Gemini 2.5 Flash permite 10 RPM — suficiente para un demo
de 2-3 preguntas.

### R5: Numeros del Word desactualizados
**Riesgo:** `evidencia_ml_herdez.docx` tiene numeros de la
meta-auditoria ($1,144,591, gap 23%) que ya no son los de produccion.
**Impacto:** Si la audiencia compara el Word con el dashboard, ve
numeros inconsistentes.
**Mitigacion:** El v3 ya esta actualizado (evidencia_ml_herdez_v3.docx).
Verificar en Bloque 7 que esta disponible como anexo.

### R6: Sobreingenieria durante Dia 4
**Riesgo:** El Dia 4 es de documentacion y narrativa, NO de codigo.
Detectar mejoras de implementacion y querer aplicarlas inmediatamente
rompe el foco y puede introducir regresiones sin tiempo para testear.
**Impacto:** Perder horas de documentacion en refactors innecesarios.
**Mitigacion:** Si se detectan mejoras de implementacion, van a
`docs/TODO_dia_5.md` como pendiente. NO modificar codigo de produccion
(src/, tests/, scripts/) durante el Dia 4.

---

## Secuencia recomendada de ejecucion

```
Bloque 1 (diagrama GCP)
    ↓
Bloque 2 (justificacion stack)
    ↓
Bloque 3 (mapeo prototipo → produccion)
    ↓
Bloque 4 (narrativa Gerente IA) ← necesita Bloques 1-3
    ↓
Bloque 5 (narrativa Director SC)
    ↓
Bloque 6 (preguntas hostiles) ← necesita Bloques 4-5
    ↓
Bloque 7 (cierre)
```

La secuencia es lineal porque cada bloque depende del anterior:
- La arquitectura GCP (1-3) informa las narrativas (4-5)
- Las narrativas (4-5) generan las preguntas hostiles (6)
- El cierre (7) documenta todo lo anterior

**Tiempo total estimado: ~8.5h.**

**Regla del Dia 4:** NO modificar codigo de produccion. Si se detectan
mejoras, van a TODO_dia_5.md.
