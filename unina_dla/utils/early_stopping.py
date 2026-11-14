import math
from typing import Any, Dict, Optional

EARLY_STOP_METRICS = ("mAP50", "mAP50-95", "fitness")


def extract_monitor_score(metrics: Dict[str, Any], metric_name: str = "mAP50") -> float:
    """Extract the scalar validation score to monitor for early stopping."""
    if metric_name == "mAP50":
        return float(metrics.get("mAP50", 0.0))
    if metric_name == "mAP50-95":
        return float(metrics.get("mAP50-95", 0.0))
    if metric_name == "fitness":
        map50 = float(metrics.get("mAP50", 0.0))
        map50_95 = float(metrics.get("mAP50-95", 0.0))
        return 0.1 * map50 + 0.9 * map50_95
    raise ValueError(
        f"Unsupported early stopping metric '{metric_name}'. "
        f"Expected one of {EARLY_STOP_METRICS}."
    )


class EarlyStopping:
    """
    Monitor a validation metric across epochs and signal when training should stop.

    Parameters
    ----------
    patience : int
        Number of consecutive epochs without sufficient improvement before stopping.
        Values <= 0 disable early stopping.
    min_delta : float
        Minimum absolute increase over ``best_score`` required to reset patience.
    min_epochs : int
        Minimum number of epochs that must complete before non-improving epochs
        begin incrementing the patience counter. Improvements during this grace
        period still update ``best_score``.
    metric_name : str
        Validation metric key to monitor (``"mAP50"``, ``"mAP50-95"``, or ``"fitness"``).
    """

    def __init__(
        self,
        patience: int = 0,
        min_delta: float = 0.0,
        min_epochs: int = 0,
        metric_name: str = "mAP50",
    ) -> None:
        if metric_name not in EARLY_STOP_METRICS:
            raise ValueError(
                f"Unsupported early stopping metric '{metric_name}'. "
                f"Expected one of {EARLY_STOP_METRICS}."
            )
        self.patience = max(0, int(patience or 0))
        self.min_delta = max(0.0, float(min_delta or 0.0))
        self.min_epochs = max(0, int(min_epochs or 0))
        self.metric_name = metric_name

        self.best_score: Optional[float] = None
        self.best_epoch: int = -1
        self.counter: int = 0
        self.should_stop: bool = False
        self.stopped_epoch: Optional[int] = None

    @property
    def enabled(self) -> bool:
        """True when early stopping is active (patience > 0)."""
        return self.patience > 0

    def step(self, epoch: int, score: float) -> bool:
        """
        Record the validation score for ``epoch`` and return True if training should stop.
        """
        if not self.enabled:
            return False

        epoch = int(epoch)
        score_val = float(score)

        if math.isfinite(score_val) and (
            self.best_score is None or score_val > (self.best_score + self.min_delta)
        ):
            self.best_score = score_val
            self.best_epoch = epoch
            self.counter = 0
            return False

        # Grace period protects warmup / curriculum transitions before counting patience.
        if (epoch + 1) <= self.min_epochs:
            self.counter = 0
            return False

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
            self.stopped_epoch = epoch
            return True
        return False

    def state_dict(self) -> Dict[str, Any]:
        """Return serializable tracking state for checkpointing."""
        return {
            "metric_name": self.metric_name,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "counter": self.counter,
            "should_stop": self.should_stop,
            "stopped_epoch": self.stopped_epoch,
        }

    def load_state_dict(self, state: Optional[Dict[str, Any]]) -> None:
        """Restore tracking state from a checkpoint dictionary."""
        if not isinstance(state, dict):
            return
        saved_metric = state.get("metric_name")
        if saved_metric is not None and saved_metric != self.metric_name:
            print(
                f"[EarlyStopping] Checkpoint tracked '{saved_metric}', current metric is "
                f"'{self.metric_name}'; resetting early stopping state."
            )
            return

        best_score = state.get("best_score")
        self.best_score = float(best_score) if best_score is not None else None
        self.best_epoch = int(state.get("best_epoch", -1))
        self.counter = max(0, int(state.get("counter", 0)))
        self.should_stop = self.enabled and (self.counter >= self.patience)
        self.stopped_epoch = state.get("stopped_epoch") if self.should_stop else None
