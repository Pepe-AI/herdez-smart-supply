# Meta-auditoría del Día 2 — Verificación de la auditoría

## Contexto

La auditoría del Día 2 concluyó que el modelo quedaba 5° de 6 baselines,
con by_stock_bajo ganando 3.2x. Esta meta-auditoría verificó la
metodología de esa auditoría y encontró dos bugs que invalidan parcialmente
los resultados.

---

## Verificación 1: Igualdad de condiciones entre baselines

### Problema encontrado: ASIMETRÍA en selección de origen

El script de la auditoría usaba una función `simulate_day_with_priority`
con selección de origen **por primer válido** (iterrows + break):

```python
origins = df_day[(df_day['sku_id'] == sku) & (df_day['cedi'] != cedi)]
for _, orig_row in origins.iterrows():
    orig_cedi = orig_row['cedi']
    if capacity_used.get(orig_cedi, 0) >= config.max_transfers_per_cedi:
        continue
    # ... toma el PRIMERO válido
    break
```

Mientras que `costs_v2.select_origin()` busca el **mejor origen**
(mayor excedente, recorre todos los candidatos):

```python
for _, row in candidates.iterrows():
    # ...
    if surplus > best_surplus:
        best_surplus = surplus
        best_origin = cedi
        best_units = transferable
```

**Impacto:** La versión corregida usa `select_origin()` para todas las
estrategias, eliminando la asimetría que penalizaba al modelo.

---

## Verificación 2: Leakage temporal en by_stock_bajo

### Problema encontrado: BUG CRÍTICO

El baseline `by_stock_bajo` usaba `stock_actual` para priorizar:

```python
stock_scores = -df_day['stock_actual'].values.astype(float)
```

Según el Diccionario de Datos del Excel:
> Stock_Actual = Inventario físico al final del día

Esto significa que `stock_actual` del día T **incluye las ventas del día T**.
Si la decisión de transferencia se toma al inicio del día T, usar
`stock_actual` de T es información del futuro.

**Corrección:** usar `stock_lag_1` (stock de ayer, disponible al momento
de decidir):

```python
stock_scores = -df_day['stock_lag_1'].values.astype(float)
```

**Impacto:** `by_stock_bajo` pasa de $400k/fold (con leakage) a
$1,644k/fold (sin leakage). Su ventaja de 3.2x era completamente
artificial.

---

## Verificación 3: Modelo usa probabilidades correctas

**Confirmado:** El modelo se reentrena por fold con TimeSeriesSplit.
Las probabilidades son out-of-fold (nunca predice sobre datos de
entrenamiento). Usa las mismas features lagged y LGBM_PARAMS del Día 1.

---

## Verificación 4: Cálculo de costos idéntico

**Confirmado:** La versión corregida (`simulate_day_fair`) usa la misma
fórmula para todas las estrategias:
- TP/FP cost: `costo_transferencia_unidad * max(deficit, 50)`
- FN cost: `costo_quiebre_stock_diario * 5`
- Alertas diferidas se cuentan como FN si y_true == 1

---

## Verificación 5: Cobertura de tests

Tests existentes que sí cubren puntos críticos:
- `test_respects_capacity` — verifica N=3 (OK)
- `test_none_when_no_safe_origin` — verifica safety (OK)
- `test_defers_negative_benefit` — verifica diferidos (OK)

Tests que **faltan**:
- Test de consistencia entre estrategias (mismos inputs → misma lógica)
- Test de que by_stock_bajo no use stock_actual
- Test end-to-end con fold real
- Tests independientes de tools.py

---

## Verificación 6: Modelo correcto

**Confirmado:** Se reentrena por fold (no usa modelo guardado del Día 1).
Parámetros idénticos (LGBM_PARAMS). Predicciones out-of-fold.

---

## Verificación 7: Reproducibilidad

El script de la auditoría original fue inline (no guardado en archivo).
La meta-auditoría se ejecutó con un nuevo script inline que corrige
ambos bugs. Random seed fijado en LGBM_PARAMS (random_state=42).
Resultados reproducibles.

---

## Ranking corregido (meta-auditoría)

TimeSeriesSplit 5 folds, N=3 transfers/día/CEDI, select_origin()
uniforme, stock_lag_1 en vez de stock_actual.

| Estrategia | Costo/fold | vs FIFO |
|---|---|---|
| by_tasa_base | $929,913 | +$794,401 |
| by_costo_quiebre | $936,815 | +$787,499 |
| **model_prioritized** | **$1,144,591** | **+$579,723** |
| heuristic_deficit | $1,522,761 | +$201,553 |
| by_stock_lag1 (corregido) | $1,644,109 | +$80,205 |
| fifo | $1,724,314 | $0 |
| inacción | $3,634,000 | -$1,909,686 |

### Comparación con auditoría original

| Estrategia | Antes (buggy) | Después (corregido) | Cambio |
|---|---|---|---|
| by_stock_bajo → by_stock_lag1 | $399,545 (1°) | $1,644,109 (5°) | +$1.2M (leakage eliminado) |
| model_prioritized | $1,294,290 (5°) | $1,144,591 (3°) | -$150k (select_origin justo) |
| by_tasa_base | $940,492 (2°) | $929,913 (1°) | -$11k (select_origin justo) |
| by_costo_quiebre | $1,007,337 (3°) | $936,815 (2°) | -$71k (select_origin justo) |

---

## Veredicto

### ¿La auditoría del Día 2 estaba metodológicamente correcta?

**No.** Dos errores materiales:
1. Asimetría en selección de origen (penalizaba al modelo)
2. Leakage temporal en by_stock_bajo (inflaba al baseline ganador)

### ¿Cambian el ranking?

**Sí, significativamente.** El modelo sube de 5° a 3°. by_stock_bajo
baja de 1° a 5°. La conclusión "el modelo es inútil" de la auditoría
original era falsa.

### ¿Algún hallazgo queda invalidado?

- **Invalidado:** "by_stock_bajo gana 3.2x" (era leakage)
- **Invalidado:** "el modelo queda 5° de 6" (era asimetría)
- **Se mantiene:** "el modelo no supera a tasa_base ni costo_quiebre"
  (confirmado con correcciones, pierde ~22%)
- **Se mantiene:** "all_positive es Pareto-óptimo bajo capacidad
  ilimitada" (no afectado por esta auditoría)

### ¿Es seguro arrancar el Día 3?

**Sí**, con esta narrativa corregida:
- El modelo es 3° de 6: supera a FIFO, stock_lag1 y heurística de
  déficit. Pierde por ~22% contra tasa_base y costo_quiebre.
- El valor del agente está en la integración de múltiples señales
  (probabilidad + costo + safety + capacidad + explicabilidad) que
  los baselines simples no hacen.
- Con más datos (12+ meses), el modelo debería mejorar; las reglas
  estáticas no.
