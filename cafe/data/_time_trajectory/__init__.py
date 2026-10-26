"""Time strategy names mapped to trajectory-building functions."""

from .base import build_trajectory_base
from .timezone import build_trajectory_timezone

TIME_STRATEGIES = {
    "base": build_trajectory_base,
    "timezone": build_trajectory_timezone,
}
