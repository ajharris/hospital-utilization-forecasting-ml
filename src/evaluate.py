from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics for forecasts."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    return {"mae": float(mae), "rmse": float(rmse)}
