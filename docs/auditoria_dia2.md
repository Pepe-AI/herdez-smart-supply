# Auditoría del Día 2 — Resultados

> **SUPERSEDIDO por `meta_auditoria_dia2.md`.** Esta auditoría contiene
> dos bugs metodológicos (asimetría en selección de origen + leakage
> temporal en by_stock_bajo) que invalidan el ranking y los costos
> reportados. Los números corregidos están en la meta-auditoría.
> Se conserva este documento como registro histórico del proceso.

## Tarea 1: Headline bajo lupa

### Código del baseline "random truncated"

El baseline `_simulate_strategy_random_truncated` en `costs_v2.py:265-308`
no es random: itera las filas del DataFrame en orden (sorted por
`[sku_id, cedi, fecha]`), lo que lo convierte en un **FIFO implícito**
por nombre de SKU. Además, intenta transferir para TODAS las filas
(no solo las con quiebre proyectado), inflando el costo de transferencia.

### Tabla comparativa: 6 baselines bajo restricción N=3

Todas las estrategias usan la misma función `simulate_day_with_priority`
que respeta N=3/CEDI y verifica safety del origen. La diferencia es
únicamente el criterio de priorización.

TimeSeriesSplit 5 folds, capacidad N=3 transferencias/día/CEDI origen.

| Estrategia | Costo/fold | vs FIFO | vs Inacción |
|---|---|---|---|
| **by_stock_bajo** | **$399,545** | **+$1,291,232** | **+$3,234,455** |
| by_tasa_base | $940,492 | +$750,285 | +$2,693,508 |
| by_costo_quiebre | $1,007,337 | +$683,440 | +$2,626,663 |
| **model_prioritized** | **$1,294,290** | **+$396,487** | **+$2,339,710** |
| heuristic_deficit | $1,484,042 | +$206,735 | +$2,149,958 |
| fifo | $1,690,777 | $0 | +$1,943,223 |
| inacción | $3,634,000 | -$1,943,223 | $0 |

Fuente: script inline ejecutado en la auditoría (no en costs_v2.py).

### Resultado

- El modelo queda **5to de 6** baselines bajo restricción N=3.
- Solo supera a FIFO y heurística de déficit.
- **Pierde contra** `by_stock_bajo` (3.2x peor), `by_tasa_base` (1.4x),
  `by_costo_quiebre` (1.3x).
- El headline "37% vs random" era contra straw-man (FIFO). No es
  defendible como número principal.

### ¿Por qué `by_stock_bajo` gana?

`stock_actual` tiene correlación -0.85 con el target. Priorizar por
menor stock es casi equivalente a predecir quiebre directamente: sin
modelo, sin features, sin entrenamiento. El modelo lagged usa
`stock_lag_1` (información de ayer), que es ligeramente peor que
`stock_actual` (información de hoy disponible al momento de decidir).

---

## Tarea 2: Lógica de capacidad

### Cómo funciona

- `capacity_used: dict[str, int]` se resetea cada día (cada llamada).
- Alertas se ordenan por `beneficio_neto` desc.
- Para cada alerta, `select_origin()` busca CEDI origen con:
  - Capacidad disponible (< 3 transfers usados)
  - Stock > safety threshold (`ventas_rolling * lead_time * 0.5`)
  - Mayor excedente entre candidatos
- Si no hay origen viable → alerta va a `deferred` (se pierde).

### Supuestos no documentados

1. **Alertas diferidas se pierden** — no se reasignan al día siguiente
   ni se intenta con otro origen alternativo.
2. **Empate en prioridad** se rompe por orden de inserción (Python
   `sorted` es estable). Impacto menor.
3. **Mínimo 50 unidades** por transferencia hardcodeado en
   `costs_v2.py:75` y `costs_v2.py:211`. Sin documentar en
   architecture.md.

### Alertas diferidas por fold (modelo)

| Fold | Alertas | Aprobadas | Diferidas | % diferido |
|------|---------|-----------|-----------|-----------|
| 1 | 162 | 108 | 54 | 33% |
| 2 | 169 | 107 | 62 | 37% |
| 3 | 164 | 103 | 61 | 37% |
| 4 | 164 | 107 | 57 | 35% |
| 5 | 164 | 106 | 58 | 35% |

~35% de alertas se difieren. Estas se convierten en FN si el quiebre
era real, lo cual infla el costo del modelo.

---

## Tarea 3: Auditoría del agente

### Nodos del grafo

| Nodo | Tipo | Función |
|------|------|---------|
| generate_alerts | Determinista | Filtra predicciones > umbral 0.0187 |
| evaluate_costs | Determinista | Filtra beneficio_neto > 0 |
| prioritize | Determinista | Greedy allocation con capacidad N=3 |
| decide | LLM | Gemini revisa el plan (no modifica) |
| explain | LLM | Gemini genera resumen en español |

### ¿Qué decisión toma el LLM que el determinista no podría?

**Ninguna decisión material.** Los nodos LLM (decide + explain) solo
generan texto explicativo. No modifican `transfers` ni `deferred`.
Toda la lógica económica se ejecuta en los nodos deterministas 1-3.

El LLM agrega:
- Explicación en lenguaje natural para el Director
- Latencia (~1-2s por llamada)
- Costo API (mínimo con Flash)

### Tools expuestas a Gemini

| Tool | Qué hace |
|------|----------|
| `obtener_alertas_hoy` | Lee alertas pre-generadas por nodo 1 |
| `verificar_seguridad_origen` | Consulta safety de un CEDI específico |
| `consultar_capacidad_restante` | Lee counter de capacidad usada |
| `ejecutar_plan_transferencias` | Valida y aprueba un plan JSON |

**Riesgo:** `ejecutar_plan_transferencias` podría aprobar un plan que
bypass la lógica determinista. No hay validación de consistencia con
lo que `prioritize_node` decidió.

**Tests de tools:** No hay tests independientes. Solo se prueban
indirectamente vía el grafo completo.

---

## Tarea 4: Auditoría de tests

### test_costs_v2.py (12 tests)

| Test | Verifica | Tipo |
|------|----------|------|
| test_filters_below_threshold | Alertas bajo umbral se filtran | Lógica económica |
| test_includes_above_threshold | Alertas sobre umbral se incluyen | Lógica económica |
| test_computes_expected_cost | p × costo × días = esperado | Lógica económica |
| test_picks_highest_surplus | Origen con mayor excedente se elige | Lógica económica |
| test_excludes_destination | Destino no puede ser su propio origen | Lógica económica |
| test_none_when_no_safe_origin | None si todos inseguros | Lógica económica |
| test_respects_capacity | None si capacidad agotada | Lógica económica |
| test_sorts_by_benefit | Mayor beneficio primero | Lógica económica |
| test_respects_n3_capacity | Ningún origen > 3 transfers | Lógica económica |
| test_defers_negative_benefit | Beneficio negativo → diferido | Lógica económica |
| test_safe_when_surplus_large | Fórmula safety (seguro) | Lógica económica |
| test_unsafe_when_surplus_small | Fórmula safety (inseguro) | Lógica económica |

### test_agent_decisions.py (6 tests)

| Test | Verifica |
|------|----------|
| test_produces_transfers_for_high_risk | Alertas altas → transferencias |
| test_no_transfers_when_all_low_risk | Sin alertas = sin transfers |
| test_respects_capacity_n3 | Ningún CEDI origen > 3 |
| test_generates_explanation | explanation no vacío |
| test_defers_when_no_origin_available | Stock bajo → diferido |
| test_high_cost_sku_prioritized | SKU $15k > SKU $8k |

### Huecos de cobertura

- **Falta:** test de ahorro contra all_positive truncado
- **Falta:** test de fold completo end-to-end (datos reales)
- **Falta:** tests independientes de tools.py
- **Falta:** test de empate en prioridad
- **Falta:** test boundary con exactamente N=3 alertas

---

## Tarea 5: Constantes y supuestos

### Constantes hardcodeadas

| Constante | Valor | Ubicación | Documentada |
|-----------|-------|-----------|-------------|
| max_transfers_per_cedi | 3 | costs_v2.py:26 | Sí (architecture.md) |
| safety_factor | 0.5 | costs_v2.py:27 | Parcial (TODO_dia_2.md item 7) |
| economic_threshold | 0.0187 | costs_v2.py:29 | Sí (threshold_sweep.py) |
| dias_expuestos | 5 | costs_v2.py:28 | Sí (CLAUDE.md) |
| **mín 50 unidades/transfer** | **50** | **costs_v2.py:75,211** | **NO** |

### Números en docstrings validados

| Número | Ubicación | Respaldo |
|--------|-----------|---------|
| ~$53,000 MXN | prompts.py:10 | scripts/cost_baselines.py |
| ~$997 MXN | prompts.py:11 | scripts/cost_baselines.py |
| 53:1 | prompts.py:12 | Derivado de anteriores |

### Supuestos no documentados en architecture.md

1. Mínimo 50 unidades por transferencia (lote logístico mínimo)
2. Alertas diferidas no se redistribuyen ni pasan al día siguiente
3. El modelo prioriza por `beneficio_neto` (p × costo - transfer),
   no por stock directo ni tasa base

---

## Veredicto

### ¿El "37% vs random" sigue siendo defendible?

**No.** Era contra FIFO (straw-man). Con baselines realistas el modelo
pierde contra 3 de 5 alternativas. La regla `min(stock_actual)` supera
al modelo por 3.2x.

### Headline corregido

No hay headline donde el modelo ML gane a todas las alternativas. El
mejor resultado honesto sería:

> "El sistema integrado (modelo + agente) reduce costos 23% vs la
> mejor heurística simple (priorización por costo de quiebre),
> incluyendo verificación de seguridad del origen y cálculo de
> volumen óptimo."

**PERO** este headline tampoco existe: $1,294,290 (modelo) es PEOR que
$1,007,337 (by_costo_quiebre). No se puede usar.

### Opción honesta

El valor del sistema no está en la predicción ni en la priorización
(donde pierde). Está en:

1. **Integración de restricciones** que los baselines simples no hacen
   (safety del origen, volumen óptimo)
2. **Explicabilidad** (el agente genera resumen para el Director)
3. **Plataforma extensible** (cuando haya más datos, el modelo mejora;
   las reglas no)

### ¿Qué rehacer antes del Día 3?

1. **Cambiar criterio de priorización** del modelo: usar `stock_actual`
   ponderado por `costo_quiebre` en vez de `beneficio_neto` basado en
   `p_quiebre` del modelo lagged.
2. **Eliminar `_simulate_strategy_random_truncated`** de costs_v2.py
   o renombrar a FIFO. Agregar los 5 baselines realistas.
3. **Documentar** mínimo 50 unidades y alertas diferidas en
   architecture.md.
4. **Pivotar narrativa**: el valor es el agente como integrador de
   decisiones, no el ML como predictor superior.
