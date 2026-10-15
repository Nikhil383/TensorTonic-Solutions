import torch

def sgns_sgd_step(W_in: torch.Tensor, W_out: torch.Tensor,
                  center_id: int, pos_id: int,
                  neg_ids: torch.Tensor, lr: float) -> dict:
    """
    Returns updated W_in and W_out float64 tensors in a dictionary.
    """
    new_W_in = W_in.clone()
    new_W_out = W_out.clone()
    center = W_in[center_id].clone()
    output_snapshot = W_out.clone()

    positive_coefficient = torch.sigmoid(torch.dot(center, output_snapshot[pos_id])) - 1.0
    center_gradient = positive_coefficient * output_snapshot[pos_id]
    output_gradients = {pos_id: positive_coefficient * center}

    for negative_id in neg_ids.tolist():
        coefficient = torch.sigmoid(torch.dot(center, output_snapshot[negative_id]))
        center_gradient = center_gradient + coefficient * output_snapshot[negative_id]
        output_gradients[negative_id] = output_gradients.get(negative_id, torch.zeros_like(center)) + coefficient * center

    new_W_in[center_id] = new_W_in[center_id] - lr * center_gradient
    for word_id, gradient in output_gradients.items():
        new_W_out[word_id] = new_W_out[word_id] - lr * gradient
    return {"W_in": new_W_in, "W_out": new_W_out}