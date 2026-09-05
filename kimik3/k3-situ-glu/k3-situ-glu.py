import torch

def situ_glu(input_tensor: torch.Tensor, gate_projection: torch.Tensor, up_projection: torch.Tensor, gate_cap: float = 4.0, up_cap: float = 25.0) -> torch.Tensor:
    """
    Returns the bounded element-wise gated activation tensor.
    """
    gate_values = input_tensor @ gate_projection.transpose(0, 1)
    up_values = input_tensor @ up_projection.transpose(0, 1)
    gate_branch = gate_cap * torch.tanh(gate_values / gate_cap) * torch.sigmoid(gate_values)
    up_branch = up_cap * torch.tanh(up_values / up_cap)
    return gate_branch * up_branch