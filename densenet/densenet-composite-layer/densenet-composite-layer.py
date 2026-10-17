import torch
import torch.nn.functional as F

def composite_layer(x: torch.Tensor, bn_gamma: torch.Tensor, bn_beta: torch.Tensor,
                    bn_mean: torch.Tensor, bn_var: torch.Tensor,
                    conv_weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Returns the float64 output of the DenseNet composite transformation.
    """
    gamma = bn_gamma[None, :, None, None]
    beta = bn_beta[None, :, None, None]
    mean = bn_mean[None, :, None, None]
    variance = bn_var[None, :, None, None]
    normalized = gamma * (x - mean) / torch.sqrt(variance + eps) + beta
    return F.conv2d(F.relu(normalized), conv_weight, bias=None, stride=1, padding=1)