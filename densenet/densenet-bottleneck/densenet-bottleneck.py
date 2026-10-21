import torch
import torch.nn.functional as F

def bottleneck_layer(x: torch.Tensor, bn1_gamma: torch.Tensor, bn1_beta: torch.Tensor,
                     bn1_mean: torch.Tensor, bn1_var: torch.Tensor,
                     conv1_weight: torch.Tensor, bn2_gamma: torch.Tensor,
                     bn2_beta: torch.Tensor, bn2_mean: torch.Tensor,
                     bn2_var: torch.Tensor, conv2_weight: torch.Tensor,
                     eps: float = 1e-5) -> torch.Tensor:
    def batch_norm(values, gamma, beta, mean, variance):
        return gamma[None, :, None, None] * (values - mean[None, :, None, None]) / torch.sqrt(variance[None, :, None, None] + eps) + beta[None, :, None, None]

    hidden = F.relu(batch_norm(x, bn1_gamma, bn1_beta, bn1_mean, bn1_var))
    hidden = F.conv2d(hidden, conv1_weight, bias=None)
    hidden = F.relu(batch_norm(hidden, bn2_gamma, bn2_beta, bn2_mean, bn2_var))
    return F.conv2d(hidden, conv2_weight, bias=None, padding=1)