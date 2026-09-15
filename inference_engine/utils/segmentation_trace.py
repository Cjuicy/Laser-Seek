from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SegmentationStages:
    """One frame's segmentation stages.

    `initial_labels` are the Felzenszwalb atoms and `merged_labels` the final labels of the
    Baseline depth method. `merge_threshold` and `depth_range` were added for IDEA-001 so that
    observation point OP-1 can report the merge criterion that was actually in force; they are
    optional so that the geometry path, which has no single depth merge threshold, can keep
    constructing this record.
    """

    initial_labels: np.ndarray
    merged_labels: np.ndarray
    confidence_threshold: float
    high_confidence_mask: np.ndarray
    merge_threshold: float | None = None
    depth_range: float | None = None


def confidence_selection(conf, quantile, method=None):
    conf = np.asarray(conf)
    if quantile is None:
        return float("nan"), np.ones(conf.shape, dtype=bool)
    kwargs = {} if method is None else {"method": method}
    threshold = float(np.quantile(conf.reshape(-1), quantile, **kwargs))
    return threshold, np.isfinite(conf) & (conf >= threshold)
