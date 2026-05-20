# TODO Dia 5 — Preparacion y ejecucion de la presentacion

## Estado al inicio del Dia 5

Sistema completo: pipeline ML, costs_v2.py con filtros uniformes,
agente LangGraph, dashboard Streamlit 3 tabs, 48 tests, 3 bugs
documentados, arquitectura GCP documentada, outline de 17 slides,
18 preguntas hostiles preparadas.

Numeros headline (produccion, fuente de verdad):
- Modelo: $1,046,657/fold (3ro de 7)
- Mejor baseline (tasa_base): $912,709/fold
- Brecha: +$133,948/fold (+14.7%)
- Ahorro vs inaccion: $2,587,343/fold (71%)

---

## Bloque 1 — Tomar screenshots del dashboard (~30 min)

**Que:** Capturar las 5 imagenes listadas en
`docs/presentation/screenshots/README.md`.

**Tareas:**
1. Levantar dashboard: `uv run streamlit run src/app/main.py`
2. Capturar con Win+Shift+S (o herramienta preferida):
   - `tab_backtest_full.png` — tab Backtest con tabla de 7 estrategias
   - `tab_backtest_chart.png` — grafico de costo por estrategia (zoom)
   - `tab_alertas_fold1.png` — tab Alertas con fold 1, decisiones y
     capacidad por CEDI
   - `tab_chat_demo.png` — tab Chat con conversacion de ejemplo
   - `dashboard_full_sidebar.png` — vista general con sidebar y metricas
3. Guardar en `docs/presentation/screenshots/` con nombres exactos.
4. Verificar que los numeros en los screenshots coinciden con los del
   outline (especialmente $1,046,657 y $912,709 en tab Backtest).

**Criterio:** 5 PNG en el directorio, numeros consistentes con outline.

**Tiempo estimado:** 30 min.

---

## Bloque 2 — Crear slides reales a partir del outline (~3h)

**Que:** Convertir `docs/presentation/outline.md` en slides
presentables.

**Tareas:**
1. Elegir formato: PowerPoint, Google Slides, o reveal.js (markdown
   renderizado).
2. Para cada uno de los 17 slides del outline:
   - Titulo conciso
   - 3-5 bullets (copiar del outline, ajustar para formato visual)
   - Insertar asset visual (screenshot, diagrama, tabla)
   - Nota al pie con referencia a evidencia del repo (para credibilidad)
3. Slide de apertura (slide 0): titulo del proyecto, nombre del
   presentador, fecha, logo Herdez si disponible.
4. Slide de cierre: resumen ejecutivo (4 parrafos del outline) +
   informacion de contacto.
5. Exportar a PDF como backup.

**Criterio:** 17+ slides con formato profesional, assets insertados.

**Tiempo estimado:** 3h.

---

## Bloque 3 — Validar el demo en vivo (~30 min)

**Que:** Verificar que el dashboard funciona end-to-end con datos
frescos.

**Tareas:**
1. `uv run streamlit run src/app/main.py`
2. Verificar tab Backtest:
   - Tabla de 7 estrategias visible y correcta
   - Grafico de costos renderiza sin errores
3. Verificar tab Alertas:
   - Selector de fold funciona
   - Tabla de alertas con decisiones coloreadas
   - Capacidad por CEDI visible
4. Verificar tab Chat:
   - Con GOOGLE_API_KEY: conversacion con Gemini real
   - Sin GOOGLE_API_KEY: mock funciona sin error
5. Si algo falla: documentar como bloqueante y decidir si se hace
   demo en vivo o se usa screenshots (Plan B).

**Criterio:** Las 3 tabs funcionan sin errores. Decision Go/No-Go
para demo en vivo.

**Tiempo estimado:** 30 min.

---

## Bloque 4 — Ensayo cronometrado (~1h)

**Que:** Practicar la presentacion completa con cronometro.

**Tareas:**
1. Preparar los slides y el outline abierto como referencia.
2. Cronometrar cada bloque:
   - Apertura: objetivo 5 min
   - Narrativa tecnica (slides 1-9): objetivo 25 min
   - Narrativa negocio (slides 10-17): objetivo 23 min
   - Cierre + Q&A: objetivo 5-7 min
3. Si algun bloque se pasa:
   - Candidatos a recortar: slide 2 (dataset, reducir a 1 min),
     slide 8 (agente, reducir a 1 min si la audiencia no es tecnica)
   - NO recortar: slide 6 (backtest, es el corazon de la evidencia),
     slide 13 (honestidad, es lo que da credibilidad)
4. Anotar transiciones que se sienten forzadas.
5. Practicar las respuestas a las 3 preguntas mas criticas:
   P1 (14.7%), P11 (falla en produccion), P17 (regla simple gana).

**Criterio:** Presentacion completa en 55-60 min. Sin slides que
se pasen de su tiempo asignado.

**Tiempo estimado:** 1h (incluyendo ajustes).

---

## Bloque 5 — Preparar respuestas a preguntas hostiles (~30 min)

**Que:** Repasar las 18 preguntas de
`docs/presentation/preguntas_hostiles.md`.

**Tareas:**
1. Leer las 18 preguntas con respuestas.
2. Memorizar las 13 de probabilidad ALTA (al menos los datos clave).
3. Practicar las respuestas en voz alta (2-3 oraciones, no mas).
4. Tener el archivo abierto durante la presentacion como referencia
   rapida en caso de pregunta inesperada.

**Criterio:** Poder responder las 13 preguntas ALTA sin consultar
el archivo.

**Tiempo estimado:** 30 min.

---

## Bloque 6 — Plan B si el demo falla (~15 min)

**Que:** Preparar fallback con screenshots.

**Tareas:**
1. Verificar que los 5 screenshots estan en
   `docs/presentation/screenshots/`.
2. Copiar screenshots a una carpeta accesible offline (USB o
   escritorio).
3. Preparar transicion verbal: "Voy a mostrarles capturas del
   dashboard en lugar del demo en vivo" (sin disculparse).
4. Verificar que `diagram_simple.png` esta accesible para el
   slide 9.

**Criterio:** Screenshots accesibles offline. Transicion ensayada.

**Tiempo estimado:** 15 min.

---

## Bloque 7 — Verificacion final pre-presentacion (~30 min)

**Que:** Checklist de entrega.

**Tareas:**
1. Verificar que `evidencia_ml_herdez.docx` (o v3) esta disponible
   como anexo entregable. Si no esta en el repo, tenerlo accesible
   por separado.
2. Repositorio limpio y pusheado:
   - `git status` sin cambios pendientes
   - `pytest tests/ -v` pasa (48 tests)
   - `ruff check src/ tests/` limpio
3. Slides exportados a PDF como backup (por si falla el proyector
   o la laptop).
4. Materiales en USB fisico como respaldo extra:
   - PDF de slides
   - Screenshots
   - evidencia_ml_herdez.docx
   - Copia del repo (zip) por si se necesita correr el dashboard
     desde otra maquina.
5. URL del repositorio lista para compartir si la audiencia la pide.

**Criterio:** Todo accesible offline. Repo pusheado. PDF de backup.

**Tiempo estimado:** 30 min.

---

## Mejoras de implementacion detectadas durante Dia 4

Estas mejoras se identificaron durante la documentacion y NO se
implementaron (regla del Dia 4: no modificar codigo de produccion).
Son candidatas para iteraciones futuras, no para el Dia 5.

1. **Fill rate como metrica de dashboard:** el outline menciona
   fill rate como metrica de negocio (CLAUDE.md convenciones) pero
   no esta calculada ni mostrada en el dashboard. Agregar como
   metrica en tab Backtest o sidebar.

2. **Encoding de caracteres en parquet:** `daily_predictions.parquet`
   muestra "Champi�ones" en lugar de "Champiñones". Verificar
   encoding al generar el parquet en `precompute_backtest.py`.

3. **Numeros exactos en tabla de backtest:** el outline usa "~$950k"
   para by_costo_quiebre. Considerar mostrar numeros exactos en
   los slides para mayor precision.

---

## Secuencia recomendada

```
Bloque 1 (screenshots) ← primero, para insertar en slides
    ↓
Bloque 2 (slides reales)
    ↓
Bloque 3 (validar demo)
    ↓
Bloque 4 (ensayo cronometrado) ← necesita slides listos
    ↓
Bloque 5 (preguntas hostiles)
    ↓
Bloque 6 (Plan B)
    ↓
Bloque 7 (verificacion final) ← ultimo paso antes de presentar
```

**Tiempo total estimado: ~6 horas** + tiempo de la presentacion.
