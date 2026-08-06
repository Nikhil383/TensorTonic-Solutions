import math
import torch


def gated_mla(
    hidden_states,
    query_projection,
    latent_down_projection,
    key_up_projection,
    value_up_projection,
    output_gate_projection,
    output_projection,
    num_heads,
    causal=True,
):
    """
    Gated Multi-head Latent Attention.

    Args:
        hidden_states:             (B, S, D)
        query_projection:          (D, D)
        latent_down_projection:    (L, D)
        key_up_projection:         (D, L)
        value_up_projection:       (D, L)
        output_gate_projection:    (D, D)
        output_projection:         (D, D)
        num_heads:                 H
        causal:                    whether to use causal masking

    Returns:
        (Y, C)

        Y: gated attention output  (B, S, D)
        C: latent KV cache         (B, S, L)
    """

    B, S, D = hidden_states.shape

    if D % num_heads != 0:
        raise ValueError(
            f"hidden dimension {D} must be divisible "
            f"by num_heads {num_heads}"
        )

    head_dim = D // num_heads

    # --------------------------------------------------
    # 1. Query projection
    #
    # Q = X W_q^T
    #
    # X:   (B, S, D)
    # W_q: (D, D)
    # Q:   (B, S, D)
    # --------------------------------------------------

    Q = hidden_states @ query_projection.T

    # --------------------------------------------------
    # 2. Compress hidden states
    #
    # C = X W_c^T
    #
    # X:   (B, S, D)
    # W_c: (L, D)
    # C:   (B, S, L)
    # --------------------------------------------------

    C = hidden_states @ latent_down_projection.T

    # --------------------------------------------------
    # 3. Reconstruct keys
    #
    # K = C W_k^T
    #
    # C:   (B, S, L)
    # W_k: (D, L)
    # K:   (B, S, D)
    # --------------------------------------------------

    K = C @ key_up_projection.T

    # --------------------------------------------------
    # 4. Reconstruct values
    #
    # V = C W_v^T
    #
    # V: (B, S, D)
    # --------------------------------------------------

    V = C @ value_up_projection.T

    # --------------------------------------------------
    # 5. Split into heads
    #
    # (B, S, D)
    # ->
    # (B, S, H, Dh)
    # ->
    # (B, H, S, Dh)
    # --------------------------------------------------

    Q = Q.reshape(
        B,
        S,
        num_heads,
        head_dim,
    ).transpose(1, 2)

    K = K.reshape(
        B,
        S,
        num_heads,
        head_dim,
    ).transpose(1, 2)

    V = V.reshape(
        B,
        S,
        num_heads,
        head_dim,
    ).transpose(1, 2)

    # Q, K, V:
    # (B, H, S, Dh)

    # --------------------------------------------------
    # 6. Attention scores
    #
    # Q K^T / sqrt(Dh)
    #
    # (B,H,S,Dh) @ (B,H,Dh,S)
    # ->
    # (B,H,S,S)
    # --------------------------------------------------

    scores = torch.matmul(
        Q,
        K.transpose(-2, -1),
    )

    scores = scores / math.sqrt(head_dim)

    # --------------------------------------------------
    # 7. Causal mask
    # --------------------------------------------------

    if causal:
        mask = torch.triu(
            torch.ones(
                S,
                S,
                device=hidden_states.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        scores = scores.masked_fill(
            mask,
            float("-inf"),
        )

    # --------------------------------------------------
    # 8. Attention probabilities
    #
    # A = softmax(scores)
    # --------------------------------------------------

    A = torch.softmax(
        scores,
        dim=-1,
    )

    # --------------------------------------------------
    # 9. Head contexts
    #
    # A @ V
    #
    # (B,H,S,S) @ (B,H,S,Dh)
    # ->
    # (B,H,S,Dh)
    # --------------------------------------------------

    context = torch.matmul(A, V)

    # --------------------------------------------------
    # 10. Concatenate heads
    #
    # (B,H,S,Dh)
    # ->
    # (B,S,H,Dh)
    # ->
    # (B,S,D)
    #
    # This is O_tilde.
    # --------------------------------------------------

    O_tilde = (
        context
        .transpose(1, 2)
        .contiguous()
        .reshape(B, S, D)
    )

    # --------------------------------------------------
    # 11. Channel gate
    #
    # G = sigmoid(X W_g^T)
    #
    # X:   (B,S,D)
    # W_g: (D,D)
    #
    # G:   (B,S,D)
    # --------------------------------------------------

    gate = torch.sigmoid(
        hidden_states @ output_gate_projection.T
    )

    # --------------------------------------------------
    # 12. Apply channel-wise gate
    #
    # G ⊙ O_tilde
    #
    # Both: (B,S,D)
    # --------------------------------------------------

    gated_output = gate * O_tilde

    # --------------------------------------------------
    # 13. Output projection
    #
    # Y = [G ⊙ O_tilde] W_o^T
    #
    # W_o: (D,D)
    #
    # Y: (B,S,D)
    # --------------------------------------------------

    Y = gated_output @ output_projection.T

    # --------------------------------------------------
    # 14. Return output and latent cache
    # --------------------------------------------------

    return Y, C