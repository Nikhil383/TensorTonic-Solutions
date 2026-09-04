import torch

_E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

def mxfp4_expert_linear(latent_tokens: torch.Tensor, packed_weights: torch.Tensor, scale_bytes: torch.Tensor, selected_experts: torch.Tensor, mixture_weights: torch.Tensor, shared_output: torch.Tensor) -> torch.Tensor:
    """
    Returns the combined routed and shared expert output tensor.
    """
    lookup = torch.tensor(_E2M1_VALUES, dtype=latent_tokens.dtype, device=latent_tokens.device)
    output = shared_output.clone()
    for token_index in range(latent_tokens.shape[0]):
        for slot in range(selected_experts.shape[1]):
            expert_index = int(selected_experts[token_index, slot])
            packed = packed_weights[expert_index]
            low = (packed & 0x0F).long()
            high = ((packed >> 4) & 0x0F).long()
            codes = torch.empty((*packed.shape[:-1], packed.shape[-1] * 2), dtype=torch.long, device=packed.device)
            codes[..., 0::2] = low
            codes[..., 1::2] = high
            values = lookup[codes]
            scales = torch.pow(
                torch.tensor(2.0, dtype=latent_tokens.dtype, device=latent_tokens.device),
                scale_bytes[expert_index].to(latent_tokens.dtype) - 127.0,
            ).unsqueeze(-1)
            weight = (values * scales).reshape(packed.shape[0], -1)
            expert_output = weight @ latent_tokens[token_index]
            output[token_index] += mixture_weights[token_index, slot] * expert_output
    return output