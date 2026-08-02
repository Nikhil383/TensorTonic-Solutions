import torch


def kda_recurrence(
    query,
    key,
    value,
    decay_logits,
    write_strength,
    output_gate_logits,
    output_projection,
    initial_state,
    g_min=-5.0,
    eps=1e-6,
):
    """
    Kimi Delta Attention recurrence.

    Returns:
        outputs:
            (B, S, model_width)

        final_state:
            (B, H, Dk, Dv)
    """

    B, S, H, Dk = query.shape
    Dv = value.shape[-1]

    # Do not mutate initial_state.
    state = initial_state.clone()

    sequence_outputs = []

    for t in range(S):
        q_t = query[:, t]              # (B, H, Dk)
        k_t = key[:, t]                # (B, H, Dk)
        v_t = value[:, t]              # (B, H, Dv)
        z_t = decay_logits[:, t]       # (B, H, Dk)

        beta_t = write_strength[:, t]

        # Normalize beta to (B, H, 1)
        if beta_t.ndim == 2:
            beta_t = beta_t.unsqueeze(-1)

        # ==================================================
        # 1. Retention
        #
        # alpha_t = exp(g_min * sigmoid(z_t))
        # ==================================================

        alpha_t = torch.exp(
            g_min * torch.sigmoid(z_t)
        )                               # (B, H, Dk)

        # Diag(alpha_t) @ S_{t-1}
        #
        # No need to explicitly construct Diag(alpha).
        retained_state = (
            alpha_t.unsqueeze(-1) * state
        )                               # (B, H, Dk, Dv)

        # ==================================================
        # 2. k_t k_t^T
        #
        # Hint 2: batched outer product
        # ==================================================

        kk_t = (
            k_t.unsqueeze(-1)
            * k_t.unsqueeze(-2)
        )                               # (B, H, Dk, Dk)

        # ==================================================
        # 3. (I - beta k k^T) retained_state
        #
        # Instead of explicitly constructing I:
        #
        # retained - beta * (k k^T @ retained)
        # ==================================================

        kk_state = torch.matmul(
            kk_t,
            retained_state
        )                               # (B, H, Dk, Dv)

        erased_state = (
            retained_state
            - beta_t.unsqueeze(-1) * kk_state
        )

        # ==================================================
        # 4. beta k v^T
        #
        # Hint 2 again.
        # ==================================================

        kv_t = (
            k_t.unsqueeze(-1)
            * v_t.unsqueeze(-2)
        )                               # (B, H, Dk, Dv)

        write = beta_t.unsqueeze(-1) * kv_t

        # ==================================================
        # 5. State update
        #
        # S_t =
        # (I - beta k k^T)
        # Diag(alpha) S_{t-1}
        # + beta k v^T
        # ==================================================

        state = erased_state + write

        # ==================================================
        # 6. Read UPDATED state
        #
        # Hint 3:
        #
        # S_t^T q_t
        # ==================================================

        out_t = (
            q_t.unsqueeze(-1) * state
        ).sum(dim=-2)                   # (B, H, Dv)

        # ==================================================
        # 7. RMS normalization
        #
        # Independently over each head's Dv values.
        # ==================================================

        rms = torch.sqrt(
            out_t.square().mean(
                dim=-1,
                keepdim=True
            ) + eps
        )

        out_t = out_t / rms

        # ==================================================
        # 8. Output gate
        # ==================================================

        gate_t = torch.sigmoid(
            output_gate_logits[:, t]
        )                               # (B, H, Dv)

        out_t = out_t * gate_t

        # ==================================================
        # 9. Concatenate heads
        #
        # (B, H, Dv)
        #      ->
        # (B, H * Dv)
        # ==================================================

        out_t = out_t.reshape(B, H * Dv)

        # ==================================================
        # 10. Output projection
        #
        # Projection is from H*Dv -> model width.
        #
        # output_projection:
        #     (model_width, H*Dv)
        # ==================================================

        out_t = out_t @ output_projection.transpose(-1, -2)

        sequence_outputs.append(out_t)

    # (B, S, model_width)
    outputs = torch.stack(
        sequence_outputs,
        dim=1
    )

    return outputs, state