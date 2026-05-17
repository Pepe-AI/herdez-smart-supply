"""Funciones puras de monetización para la matriz de confusión.

Cada celda de la matriz se traduce a un impacto económico en MXN,
permitiendo evaluar el modelo en términos de negocio.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class MonetizedMatrix:
    """Resultado de monetizar la matriz de confusión."""

    # Costos totales por celda (MXN)
    tp_cost: float  # Costo de transferencias correctas (acción tomada, quiebre evitado)
    fp_cost: float  # Costo de transferencias innecesarias (acción sin beneficio)
    fn_cost: float  # Costo de quiebres no detectados (inacción con consecuencia)
    tn_cost: float  # Sin costo (inacción correcta)

    # Conteos
    tp_count: int
    fp_count: int
    fn_count: int
    tn_count: int

    # Baseline y ahorro (calculados en monetized_confusion_matrix)
    cost_no_model: float  # Costo si no existiera modelo (todos los quiebres ocurren)
    savings_vs_no_model: float  # cost_no_model - cost_with_model

    @property
    def total_error_cost(self) -> float:
        """Costo total de errores del modelo (FP + FN)."""
        return self.fp_cost + self.fn_cost


def monetized_confusion_matrix(
    y_true: npt.NDArray[np.int_],
    y_pred: npt.NDArray[np.int_],
    costo_quiebre_diario: npt.NDArray[np.float64],
    costo_transferencia: npt.NDArray[np.float64],
    dias_expuestos: int = 5,
) -> MonetizedMatrix:
    """Calcula la matriz de confusión monetizada en MXN.

    Args:
        y_true: Labels reales (1=quiebre, 0=no quiebre).
        y_pred: Predicciones del modelo (1=quiebre, 0=no quiebre).
        costo_quiebre_diario: Costo por día de quiebre por cada observación.
        costo_transferencia: Costo de transferir una unidad por observación.
        dias_expuestos: Días de exposición al riesgo (ventana de predicción).

    Returns:
        MonetizedMatrix con costos desglosados por celda.
    """
    # Máscaras booleanas para cada celda de la matriz
    tp_mask = (y_true == 1) & (y_pred == 1)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    tn_mask = (y_true == 0) & (y_pred == 0)

    # TP: transferimos y había quiebre → pagamos costo de transferencia
    # (pero evitamos el costo de quiebre; el "beneficio" es implícito)
    tp_cost = float(np.sum(costo_transferencia[tp_mask]))

    # FP: transferimos pero no había quiebre → costo de transferencia sin beneficio
    fp_cost = float(np.sum(costo_transferencia[fp_mask]))

    # FN: no actuamos y hubo quiebre → sufrimos costo de quiebre por días expuestos
    fn_cost = float(np.sum(costo_quiebre_diario[fn_mask] * dias_expuestos))

    # TN: no actuamos y no hubo quiebre → costo cero
    tn_cost = 0.0

    # Baseline: sin modelo, TODOS los quiebres reales causan su costo
    cost_no_model = float(np.sum(costo_quiebre_diario[y_true == 1] * dias_expuestos))

    # Con modelo: pagamos transferencias (TP+FP) + quiebres no detectados (FN)
    cost_with_model = tp_cost + fp_cost + fn_cost

    return MonetizedMatrix(
        tp_cost=tp_cost,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
        tn_cost=tn_cost,
        tp_count=int(np.sum(tp_mask)),
        fp_count=int(np.sum(fp_mask)),
        fn_count=int(np.sum(fn_mask)),
        tn_count=int(np.sum(tn_mask)),
        cost_no_model=cost_no_model,
        savings_vs_no_model=cost_no_model - cost_with_model,
    )
