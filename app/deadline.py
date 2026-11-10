"""Per-request time budget shared by every stage of one request."""
import time

from . import config


class Deadline:
    def __init__(self, seconds: float):
        self._end = time.monotonic() + seconds

    def remaining(self) -> float:
        """Seconds left for a new call, after keeping the safety margin."""
        return max(0.0, self._end - time.monotonic() - config.DEADLINE_MARGIN_S)

    def cap(self, seconds: float) -> float:
        """A call timeout that never runs past the request budget."""
        return min(seconds, self.remaining())
