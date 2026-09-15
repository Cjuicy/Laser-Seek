"""Top-confidence selection, shared by all three segmentation methods.

Semantics
---------
`confidence_keep_ratio` is the **complement of the retained fraction**: a pixel is kept when

    conf >= quantile(conf[finite], 1 - keep_ratio)

so `keep_ratio = 0.7` keeps roughly the top 30% of pixels, and `keep_ratio = 0.5` keeps roughly
the top half.

This is the Reference implementation's convention, and it is also exactly the Baseline's: the
streaming engine has always computed `1 - top_conf_percentile` in its constructor and then used
that value as the quantile level, so a historical `--top_conf_percentile 0.3` is
`confidence_keep_ratio = 0.7` and keeps the top 30%. The IDEA-001 Regression gate requires the
depth path to stay element-wise identical, and this convention satisfies it.

The name is kept because the Reference uses it and a comparison must be able to quote one
parameter. The value is *not* a fraction of pixels to keep; the shipped configuration says 0.7
precisely because it must reproduce the historical `0.3` cut.

Non-finite confidence values are excluded before the quantile is taken. That is stricter than the
Baseline's implicit `NaN >= threshold == False`, and produces the same selection whenever the
confidence map has no non-finite entries.
"""

from __future__ import annotations

import numpy as np

# Deliberately narrower than NumPy's full set. `numpy.quantile` accepts interpolation methods such
# as "linear" and "midpoint", but a confidence CUT should be a value that actually occurs in the
# data, so only the two order-statistic rules are allowed. This also matches the Reference
# implementation, whose `ConfidenceQuantileMethod` enum offers exactly these two.
SUPPORTED_METHODS = ("higher", "nearest")


def select_numpy_top_confidence_mask(
    confidence: np.ndarray,
    keep_ratio: float,
    method: str = "nearest",
) -> np.ndarray:
    """Boolean mask of the top-confidence pixels.

    `keep_ratio` is the complement of the retained fraction, i.e. the quantile level used is
    `1 - keep_ratio`. Values equal to the threshold stay selected, matching the Baseline.
    """
    values = np.asarray(confidence)
    if not isinstance(method, str) or method not in SUPPORTED_METHODS:
        raise ValueError(
            f"confidence quantile method must be one of {SUPPORTED_METHODS}, got {method!r}"
        )
    if not np.isfinite(keep_ratio) or not 0.0 < float(keep_ratio) <= 1.0:
        raise ValueError("confidence keep_ratio must be in (0, 1]")

    finite = np.isfinite(values)
    finite_values = values[finite]
    if finite_values.size == 0:
        raise ValueError("confidence contains no finite values")

    threshold = np.quantile(finite_values, 1.0 - float(keep_ratio), method=method)
    return finite & (values >= threshold)


__all__ = ["select_numpy_top_confidence_mask", "SUPPORTED_METHODS"]
