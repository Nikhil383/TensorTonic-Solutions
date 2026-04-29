def detect_drift(reference_counts, production_counts, threshold):
    """
    Compare reference and production distributions to detect data drift.
    """
    # Write code here
    ref_total = sum(reference_counts)
    prod_total = sum(production_counts)

    ref = [x / ref_total for x in reference_counts]
    prod = [x / prod_total for x in production_counts]

    # Compute TVD
    tvd = 0.5 * sum(abs(r - p) for r, p in zip(ref, prod))

    return {
        "score": float(tvd),
        "drift_detected": tvd > threshold
    }