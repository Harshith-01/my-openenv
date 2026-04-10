from typing import Final


GRADER_EPSILON: Final[float] = 0.01


def to_open_interval(value: float, epsilon: float = GRADER_EPSILON) -> float:
    """Map any [0, 1] value to strict (0, 1)."""
    clamped = max(0.0, min(1.0, float(value)))
    return epsilon + clamped * (1.0 - (2.0 * epsilon))
