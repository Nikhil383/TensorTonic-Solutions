from math import ceil

def evaluate_shadow(production_log, shadow_log, criteria):
    """
    Evaluate whether a shadow model is ready for promotion.
    """

    # Compute accuracies
    production_accuracy = sum(
        entry["prediction"] == entry["actual"]
        for entry in production_log
    ) / len(production_log)

    shadow_accuracy = sum(
        entry["prediction"] == entry["actual"]
        for entry in shadow_log
    ) / len(shadow_log)

    accuracy_gain = shadow_accuracy - production_accuracy

    # Compute P95 latency (nearest-rank method)
    latencies = sorted(entry["latency_ms"] for entry in shadow_log)
    n = len(latencies)
    p95_index = ceil(0.95 * n) - 1
    shadow_latency_p95 = latencies[p95_index]

    # Compute agreement rate
    production_preds = {entry["input_id"]: entry["prediction"] for entry in production_log}
    shadow_preds = {entry["input_id"]: entry["prediction"] for entry in shadow_log}

    common_ids = production_preds.keys() & shadow_preds.keys()

    agreement_count = sum(
        production_preds[i] == shadow_preds[i]
        for i in common_ids
    )

    agreement_rate = agreement_count / len(common_ids) if common_ids else 0.0

    # Promotion decision
    promote = (
        accuracy_gain >= criteria["min_accuracy_gain"]
        and shadow_latency_p95 <= criteria["max_latency_p95"]
        and agreement_rate >= criteria["min_agreement_rate"]
    )

    return {
        "promote": promote,
        "metrics": {
            "shadow_accuracy": shadow_accuracy,
            "production_accuracy": production_accuracy,
            "accuracy_gain": accuracy_gain,
            "shadow_latency_p95": shadow_latency_p95,
            "agreement_rate": agreement_rate,
        },
    }