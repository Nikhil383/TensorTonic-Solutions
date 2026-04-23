import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    # Write code here
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    # Total sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    
    # Residual sum of squares
    ss_res = np.sum((y_true - y_pred)**2)
    
    # Handle constant target case
    if ss_tot == 0:
        return 1.0 if np.allclose(y_true, y_pred) else 0.0
    
    # R² formula
    r2 = 1 - (ss_res / ss_tot)
    return r2