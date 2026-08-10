import torch


def full_attention_residual(
    embedding,
    previous_outputs,
    pseudo_query,
    eps=1e-6
):
    """
    Full attention over the embedding and previous outputs.

    Parameters
    ----------
    embedding : torch.Tensor
        Shape: [batch, seq_len, dim]

    previous_outputs : torch.Tensor
        Shape: [depth, batch, seq_len, dim]

    pseudo_query : torch.Tensor
        Shape: [dim]

    eps : float
        Numerical stability term.

    Returns
    -------
    retrieved : torch.Tensor
        Shape: [batch, seq_len, dim]

    attention_weights : torch.Tensor
        Shape: [depth + 1, batch, seq_len]
    """

    # ---------------------------------------------------------
    # 1. Add embedding as the first depth source
    # ---------------------------------------------------------
    sources = torch.cat(
        (
            embedding.unsqueeze(0),
            previous_outputs
        ),
        dim=0
    )

    # sources:
    # [depth + 1, batch, seq_len, dim]

    # ---------------------------------------------------------
    # 2. RMS normalization
    # ---------------------------------------------------------
    rms_scale = torch.sqrt(
        sources.square().mean(
            dim=-1,
            keepdim=True
        ) + eps
    )

    normalized_sources = sources / rms_scale

    # ---------------------------------------------------------
    # 3. Compute attention logits
    # ---------------------------------------------------------
    #
    # normalized_sources:
    # [depth, batch, seq_len, dim]
    #
    # pseudo_query:
    # [dim]
    #
    # Broadcasting gives:
    # [depth, batch, seq_len]
    #
    logits = (
        normalized_sources * pseudo_query
    ).sum(dim=-1)

    # ---------------------------------------------------------
    # 4. Normalize attention across depth
    # ---------------------------------------------------------
    attention_weights = torch.softmax(
        logits,
        dim=0
    )

    # ---------------------------------------------------------
    # 5. Weighted sum of ORIGINAL sources
    # ---------------------------------------------------------
    retrieved = (
        attention_weights.unsqueeze(-1) * sources
    ).sum(dim=0)

    return retrieved, attention_weights