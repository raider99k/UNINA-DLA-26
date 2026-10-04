import math


def _calculate_cosine_schedule(epoch: int, total_epochs: int, start: float, end: float) -> float:
    """
    Generic cosine schedule from `start` to `end`.
    """
    if total_epochs <= 0:
        return start

    progress = (epoch + 1) / total_epochs
    cosine_value = (1 - math.cos(math.pi * progress)) / 2
    return start + (end - start) * cosine_value


def calculate_teacher_fade(
    epoch: int, total_epochs: int, min_fade: float = 0.0, max_fade: float = 0.1
) -> float:
    """
    Calculate Teacher Fade factor with cosine warmup for smooth ramp.

    Teacher Fade is a regularization term that penalizes reliance on distillation
    losses as training progresses. This creates a self-regulating decay that
    adapts to dataset difficulty rather than following a blind schedule.

    Args:
        epoch: Current epoch (0-indexed)
        total_epochs: Total epochs for this phase
        min_fade: Minimum fade penalty at start of phase (default 0.0)
        max_fade: Maximum fade penalty at end of phase (default 0.1)

    Returns:
        fade_factor: Value in [min_fade, max_fade] following cosine ramp (S-curve)

    Reference: Cost-Aware Uncertainty Balancing for Knowledge Distillation
    """
    return _calculate_cosine_schedule(
        epoch=epoch, total_epochs=total_epochs, start=min_fade, end=max_fade
    )


def calculate_task_weight(
    epoch: int, total_epochs: int, start_weight: float = 0.5, end_weight: float = 1.0
) -> float:
    """
    Calculate task-loss weight using an inverse-fade cosine ramp.

    This is the complementary schedule to Teacher Fade: task loss starts lower
    (teacher-dominant regime) and rises over time toward full GT authority.

    Args:
        epoch: Current epoch (0-indexed)
        total_epochs: Total epochs for this phase
        start_weight: Task-loss weight at phase start
        end_weight: Task-loss weight at phase end

    Returns:
        task_weight: Value in [start_weight, end_weight] following cosine ramp
    """
    return _calculate_cosine_schedule(
        epoch=epoch, total_epochs=total_epochs, start=start_weight, end=end_weight
    )


def calculate_component_weight(
    epoch: int, total_epochs: int, start_weight: float = 1.0, end_weight: float = 1.0
) -> float:
    """
    Generic cosine schedule for per-component loss weighting.

    Useful for independently shaping KD vectors (e.g., keep SDF high longer while
    fading logit/box faster) without changing deployment behavior.
    """
    return _calculate_cosine_schedule(
        epoch=epoch, total_epochs=total_epochs, start=start_weight, end=end_weight
    )
