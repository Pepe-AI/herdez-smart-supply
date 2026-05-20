# TODO Dia 4 — Presentacion + Deploy

## Prioridad 0: actualizar evidencia

0. **Actualizar Word v2 con numeros corregidos**: el documento
   `evidencia_ml_herdez.docx` tiene los numeros de la meta-auditoria
   ($1,144,591, $929,913, gap 23%). Los numeros de produccion
   verificados son: modelo $1,046,657, tasa_base $912,709,
   gap +14.7%. Actualizar todas las tablas y narrativa.

## Prioridad 1: mejorar el dashboard

1. **Fill rate y ROI en sidebar**: calcular fill rate
   (transferencias que evitaron quiebre / quiebres totales) y ROI
   (ahorro en quiebres evitados / costo de transferencias).
   Deben aparecer en el sidebar como metricas headline.

2. **Mejorar tab Chat**: actualmente ejecuta el agente por dia,
   pero no tiene contexto conversacional real. Agregar:
   - Historial de conversacion que persista entre recargas
   - Pregunta tipo "por que se difirió X?" con contexto

3. **Activar Gemini real**: copiar GOOGLE_API_KEY de .env.example
   a .env y verificar que el chat genera respuestas en español.
   Correr el test de equivalencia decisional.

## Prioridad 2: tests faltantes (meta-auditoria)

4. **test_tools.py**: tests independientes de src/agent/tools.py
   (obtener_alertas_hoy, verificar_seguridad_origen, etc).

5. **Test end-to-end con fold real**: cargar datos de DuckDB,
   ejecutar run_capacity_comparison con 1 fold, verificar que
   los costos estan en rango esperado.

## Prioridad 3: presentacion

6. **Screenshots del dashboard**: capturar los 3 tabs para
   incluir en la presentacion/slides.

7. **Narrativa de la presentacion**:
   - Problema: quiebres de stock cuestan $3.6M/fold sin accion
   - Solucion: modelo ML + agente economico con restricciones
   - Resultado: 3ro de 7 baselines, gap de 14.7% vs mejor regla
   - Valor del agente: integra multiples senales (p, costo, safety,
     capacidad) que las reglas simples no combinan
   - Con mas datos (12+ meses), el modelo mejora; las reglas no

## Prioridad 4: deploy (opcional)

8. **Render o Streamlit Cloud**: deploy basico del dashboard.
   Solo si hay tiempo despues de presentacion.

## Decision resuelta (Dia 3 debugging)

9. **Datos para la presentacion**: usar numeros de produccion
   (costs_v2.py). Son reproducibles, testeados, y aplican filtros
   uniformes. La meta-auditoria usó un script inline no guardado
   y es irreproducible. Numeros de produccion son la fuente de verdad.
