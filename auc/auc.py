import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    # Write code here
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    
    # Validation
    if fpr.shape != tpr.shape:
        raise ValueError("fpr and tpr must have the same length")
    if fpr.size < 2:
        raise ValueError("At least two points are required")
    
    # Ensure correct ordering along x-axis (FPR)
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]
    
    # Trapezoidal integration
    auc_value = np.trapezoid(tpr, fpr)
    
    return float(auc_value)