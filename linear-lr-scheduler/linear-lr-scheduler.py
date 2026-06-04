def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0) -> float:
    """
    Linear warmup (0→initial_lr) then linear decay (initial_lr→final_lr).
    Steps are 0-based; clamp at final_lr after total_steps.
    """
    # Write code here
    # Warmup phase
    if warmup_steps > 0 and step < warmup_steps:
        return initial_lr * (step / warmup_steps)

    # After training is finished
    if step >= total_steps:
        return final_lr

    # Decay phase
    decay_steps = total_steps - warmup_steps

    if decay_steps <= 0:
        return final_lr

    progress = (step - warmup_steps) / decay_steps
    lr = initial_lr + progress * (final_lr - initial_lr)

    return lr