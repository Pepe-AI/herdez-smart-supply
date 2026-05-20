# Arquitectura del Agente Economico

## Restriccion de capacidad logistica

**Parametro de diseno:** maximo N=3 transferencias diarias por CEDI origen.

### Justificacion operativa

Cada transferencia entre CEDIs requiere:
1. Coordinacion entre almacen origen y destino (~30 min)
2. Picking y preparacion de la carga (~45 min)
3. Carga y despacho (~30 min)
4. Documentacion y actualizacion de inventario (~15 min)

Total: ~2 horas por transferencia. Con jornada operativa de 8 horas y
overlap parcial entre procesos, un CEDI puede ejecutar maximo 3
transferencias simultaneas sin disrumpir su operacion normal de despacho
a clientes.

Con 4 CEDIs: **maximo 12 transferencias diarias en toda la red.**

### Por que esta restriccion cambia el valor del ML

Sin restriccion: transferir siempre (all_positive) cuesta $292k/fold y
es optimo. No necesitas modelo.

Con restriccion de 12 transferencias/dia sobre ~9 dias/fold:
- Capacidad maxima: 108 transferencias por fold
- all_positive necesita: 176 transferencias por fold
- **Deficit: 68 transferencias que all_positive no puede ejecutar**

Cuando hay mas alertas que capacidad, el sistema debe PRIORIZAR.
Un sistema sin modelo prioriza arbitrariamente (FIFO, por SKU, etc).
El modelo asigna probabilidades calibradas que permiten priorizar por
**costo esperado de quiebre**: `p_quiebre * costo_quiebre_diario * dias`.

## Flujo del agente

```
Cada dia a las 06:00:

1. GENERAR ALERTAS
   - Ejecutar modelo LAGGED sobre inventario de ayer
   - Calcular p_quiebre para cada SKU-CEDI
   - Filtrar: alertas donde p > umbral_economico (0.0187)

2. EVALUAR COSTOS
   Para cada alerta:
   - costo_esperado_no_actuar = p_quiebre * costo_quiebre_diario * dias_expuestos
   - costo_transferir = costo_unidad * deficit_estimado
   - beneficio_neto = costo_esperado_no_actuar - costo_transferir
   Filtrar: solo alertas con beneficio_neto > 0

3. PRIORIZAR (aqui esta el valor del agente)
   - Ordenar alertas por beneficio_neto descendente
   - Para cada CEDI origen: seleccionar top-N=3 alertas
   - Verificar que el CEDI origen no quede en riesgo tras la transferencia

4. DECIDIR
   Para cada transferencia seleccionada:
   - Determinar CEDI origen (el que tenga mayor excedente del SKU)
   - Calcular unidades a mover: min(deficit_destino, excedente_origen * 0.5)
   - Si excedente_origen < umbral_seguridad: no transferir, esperar reabasto

5. EXPLICAR
   - Generar resumen en lenguaje natural para el Director de Supply Chain
   - Incluir: que se movio, de donde a donde, por que, cuanto cuesta,
     cuanto se ahorra vs no actuar
```

## Donde aporta cada componente

| Componente | Valor | Sin este componente |
|---|---|---|
| Modelo ML | Probabilidades calibradas por SKU-CEDI-dia | Priorizar arbitrariamente (FIFO) |
| Agente economico | Decision de costo-beneficio con restricciones | Transferir todo sin criterio |
| Restriccion de capacidad | Fuerza priorizacion, da sentido al modelo | all_positive es optimo trivial |
| Dashboard | Visibilidad para el Director | Decisiones opacas |

## Diseno del agente economico (Dia 2)

### Trigger

Alerta del modelo con probabilidad > umbral economico (0.0187).
Se ejecuta diariamente sobre el inventario del dia anterior.

### Inputs del agente

- Probabilidades del modelo LAGGED (por SKU-CEDI-fecha).
- Stocks actuales y proyectados de todos los CEDIs (via DuckDB).
- Costos: transferencia por unidad por origen-destino, quiebre por SKU.
- Capacidad logistica disponible por CEDI origen (parametro N=3).
- Umbral de seguridad del stock origen (pendiente definir Dia 2).

### Outputs del agente

- Lista priorizada de transferencias a ejecutar hoy.
- Lista de alertas diferidas para evaluacion al dia siguiente.
- Razonamiento textual (Gemini 2.5 Flash) para cada decision.

### Decisiones del agente

1. **Priorizar**: ordenar alertas por valor esperado en riesgo
   = `p_quiebre * costo_quiebre_diario * dias_exposicion`.
2. **Seleccionar origen**: CEDI con mayor excedente del SKU que no
   quede en riesgo tras la transferencia.
3. **Calcular volumen**: `min(deficit_destino, excedente_origen * 0.5)`.
4. **No actuar cuando**: capacidad agotada del CEDI origen, o ningun
   origen viable (todos en riesgo para ese SKU).

### Política de filtros del backtest

Todas las estrategias del backtest (`run_capacity_comparison`) aplican
dos filtros antes de priorizar:

1. **Filtro de umbral económico** (p_quiebre > 0.0187): derivado del
   régimen de costos 53:1. Representa "¿vale la pena actuar
   económicamente?". La probabilidad p viene del modelo ML para todas
   las estrategias, incluso las que no usan p para priorizar.

2. **Filtro de beneficio neto positivo**: costo_esperado_no_actuar >
   costo_transferencia. Representa "¿el costo de actuar es menor que
   el costo esperado de no actuar?".

Las 6 estrategias activas se diferencian únicamente en el criterio
de priorización del conjunto ya filtrado. Esta política garantiza que
la comparación sea justa y que ninguna estrategia tome decisiones
económicamente irracionales.

Decisión documentada el 2026-05-19, debugging Día 3 fase 2.
