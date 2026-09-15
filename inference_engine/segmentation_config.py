"""Single source of truth for segmentation parameters across every entry point.

IDEA-001 (`docs/dsh/ideas/IDEA-001-segmentation-method-attribution.md`) requires that a
comparison of initial segmentation methods vary exactly one thing: the method. Before this
module existed, each entry point hard-coded its own window size, overlap, confidence ratio and
`depth_refine` flag, so results from different entry points were not comparable and no run
recorded which values were actually in force.

Two rules keep this module honest:

1. Values here are frozen once a run starts; `SegmentationConfig` is immutable and
   `resolve_config()` is the only constructor used by entry points.
2. Observation configuration (`DiagnosticsConfig`) is deliberately separate. Changing how much
   is observed must never change the experiment condition.

`method` is the only field intended to differ between the runs of one comparison.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

METHOD_DEPTH = "depth"
METHOD_GEOMETRY = "geometry"
METHOD_ATOMIC = "atomic"
SEGMENTATION_METHODS = (METHOD_DEPTH, METHOD_GEOMETRY, METHOD_ATOMIC)

DEFAULT_CONFIG_PATH = "configs/segmentation_config.yaml"
ENV_CONFIG_PATH = "LASER_SEGMENTATION_CONFIG"
ENV_METHOD = "LASER_SEGMENTATION_METHOD"
ENV_DIAGNOSTICS = "LASER_SEGMENTATION_DIAGNOSTICS"
ENV_DIAGNOSTICS_DIR = "LASER_SEGMENTATION_DIAGNOSTICS_DIR"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class FelzenszwalbConfig:
    """Initial Felzenszwalb segmentation parameters, locked for all three methods.

    IDEA-001 decision: all three methods use 300 / 1.1 / 500. The Reference implementation
    tunes `geometry` to 200 / 1.0 / 300; inheriting that would mix a parameter change into the
    method comparison and destroy attribution. The consequence is recorded in the run identity:
    the `geometry` numbers produced here are *not* the Reference's tuned configuration.
    """

    scale: float = 300.0
    sigma: float = 1.1
    min_size: int = 500

    def __post_init__(self) -> None:
        _require(self.scale > 0, "felzenszwalb.scale must be positive")
        _require(self.sigma > 0, "felzenszwalb.sigma must be positive")
        _require(int(self.min_size) >= 1, "felzenszwalb.min_size must be >= 1")


@dataclass(frozen=True)
class GeometryConfig:
    """Geometry-aware merge criteria (method=`geometry`, and the split stage of `atomic`)."""

    normal_method: str = "cross"
    normal_threshold_degrees: float = 20.0

    def __post_init__(self) -> None:
        _require(
            self.normal_method in ("cross", "sobel", "pca"),
            f"unsupported normal_method: {self.normal_method!r}",
        )
        _require(
            0.0 < float(self.normal_threshold_degrees) <= 90.0,
            "geometry.normal_threshold_degrees must be in (0, 90]",
        )


@dataclass(frozen=True)
class AtomicConfig:
    """Layer-atomic merge/split parameters (method=`atomic`)."""

    split_mode: str = "conservative"
    split_score_threshold: float = 0.10

    def __post_init__(self) -> None:
        _require(
            self.split_mode in ("none", "conservative", "normal_only"),
            f"unsupported atomic.split_mode: {self.split_mode!r}",
        )
        _require(
            float(self.split_score_threshold) >= 0.0,
            "atomic.split_score_threshold must be >= 0",
        )


@dataclass(frozen=True)
class IRLSConfig:
    """Per-layer scale estimation and camera scale estimation solvers.

    These belong to LSA, not to segmentation: all three methods share them, so they are locked.
    """

    iters: int = 10
    eps: float = 1e-8
    stop_tol: float = 0.05
    clamp_min: float = 1e-6

    def __post_init__(self) -> None:
        _require(int(self.iters) >= 1, "irls.iters must be >= 1")
        _require(float(self.eps) > 0, "irls.eps must be positive")
        _require(float(self.stop_tol) > 0, "irls.stop_tol must be positive")
        _require(float(self.clamp_min) > 0, "irls.clamp_min must be positive")


@dataclass(frozen=True)
class SegmentationConfig:
    """Every parameter that may influence which regions reach the LSA stage.

    `method` is the only intended independent variable of the IDEA-001 comparison.
    """

    method: str = METHOD_DEPTH

    # Confidence selection. `confidence_keep_ratio` is the complement of the retained fraction:
    # pixels are kept when `conf >= quantile(conf, 1 - confidence_keep_ratio)`, so 0.7 keeps the
    # top 30%. This is the Reference implementation's convention AND the Baseline's, whose engine
    # has always passed `1 - top_conf_percentile` as the quantile level. A historical
    # `--top_conf_percentile 0.3` therefore maps to `confidence_keep_ratio = 0.7`.
    confidence_keep_ratio: float = 0.7
    confidence_quantile_method: str = "nearest"

    # Region correspondence thresholds. Two different values are used by the Baseline itself:
    # intra-frame matching inside one window, and cross-window (inter) matching against the
    # anchor window. IDEA-001 keeps both Baseline values rather than "fixing" them.
    corr_iou_thresh_intra: float = 0.3
    corr_iou_thresh_inter: float = 0.4

    depth_merge_thresh: float = 0.1

    felzenszwalb: FelzenszwalbConfig = field(default_factory=FelzenszwalbConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    atomic: AtomicConfig = field(default_factory=AtomicConfig)
    irls: IRLSConfig = field(default_factory=IRLSConfig)

    # Engine-side knobs that must be identical across every entry point of a comparison.
    window_size: int = 20
    overlap: int = 5
    depth_refine: bool = True
    sample_interval: int = 1

    def __post_init__(self) -> None:
        _require(
            self.method in SEGMENTATION_METHODS,
            f"unsupported segmentation method: {self.method!r}; "
            f"expected one of {SEGMENTATION_METHODS}",
        )
        _require(
            bool(self.confidence_keep_ratio)
            and 0.0 < float(self.confidence_keep_ratio) <= 1.0,
            "confidence_keep_ratio must be in (0, 1]",
        )
        _require(
            self.confidence_quantile_method in ("nearest", "higher", "lower", "midpoint",
                                                "linear"),
            f"unsupported confidence_quantile_method: {self.confidence_quantile_method!r}",
        )
        _require(
            0.0 <= float(self.corr_iou_thresh_intra) <= 1.0,
            "corr_iou_thresh_intra must be in [0, 1]",
        )
        _require(
            0.0 <= float(self.corr_iou_thresh_inter) <= 1.0,
            "corr_iou_thresh_inter must be in [0, 1]",
        )
        _require(
            float(self.depth_merge_thresh) >= 0.0,
            "depth_merge_thresh must be >= 0",
        )
        _require(int(self.window_size) >= 1, "window_size must be >= 1")
        _require(
            1 <= int(self.overlap) < int(self.window_size),
            "overlap must satisfy 1 <= overlap < window_size",
        )
        _require(int(self.sample_interval) >= 1, "sample_interval must be >= 1")

    # -- derived helpers ------------------------------------------------------------------

    def window_step(self) -> int:
        return int(self.window_size) - int(self.overlap)

    def with_method(self, method: str) -> "SegmentationConfig":
        """Return a copy with only `method` replaced — the single IDEA-001 independent variable."""
        return dataclasses.replace(self, method=method)

    def run_identity(self) -> dict[str, Any]:
        """Everything needed to reproduce a run and to attribute a difference to the method."""
        return {
            "method": self.method,
            "window_size": int(self.window_size),
            "overlap": int(self.overlap),
            "depth_refine": bool(self.depth_refine),
            "sample_interval": int(self.sample_interval),
            "confidence_keep_ratio": float(self.confidence_keep_ratio),
            "confidence_quantile_method": self.confidence_quantile_method,
            "corr_iou_thresh_intra": float(self.corr_iou_thresh_intra),
            "corr_iou_thresh_inter": float(self.corr_iou_thresh_inter),
            "depth_merge_thresh": float(self.depth_merge_thresh),
            "felzenszwalb": dataclasses.asdict(self.felzenszwalb),
            "geometry": dataclasses.asdict(self.geometry),
            "atomic": dataclasses.asdict(self.atomic),
            "irls": dataclasses.asdict(self.irls),
        }

    def run_identity_hash(self) -> str:
        import hashlib

        payload = json.dumps(self.run_identity(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class DiagnosticsConfig:
    """Observation configuration. Separate from `SegmentationConfig` on purpose.

    Changing the observation level must never change the experiment condition, so this object is
    never consulted by any computation that produces labels, scales or point maps.
    """

    enabled: bool = True
    level: str = "scalar"
    output_dir: str | None = None
    dump_arrays: bool = False

    def __post_init__(self) -> None:
        _require(
            self.level in ("off", "scalar", "arrays"),
            f"unsupported diagnostics level: {self.level!r}",
        )
        _require(
            not self.dump_arrays or self.level == "arrays",
            "dump_arrays requires level='arrays'",
        )

    @property
    def active(self) -> bool:
        return bool(self.enabled) and self.level != "off"

    @property
    def wants_arrays(self) -> bool:
        return self.active and self.level == "arrays"


def _parse_scalar(value: str) -> Any:
    lowered = value.strip().strip('"').strip("'")
    if lowered.lower() in ("true", "false"):
        return lowered.lower() == "true"
    if lowered.lower() in ("none", "null", "~"):
        return None
    try:
        return int(lowered)
    except ValueError:
        pass
    try:
        return float(lowered)
    except ValueError:
        pass
    return lowered


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Read a flat two-level YAML mapping without adding a dependency.

    The shipped config file uses only nested mappings and scalars, which is all this reader
    handles. Any other structure is a hard error rather than a silent mis-parse.
    """
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if ":" not in line:
            raise ValueError(f"{path}:{lineno}: expected 'key: value', got {line!r}")
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"{path}:{lineno}: inconsistent indentation")
        parent = stack[-1][1]
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


# Every override is addressed from the file root, so the section name is part of the path. This
# is easy to get wrong: an override keyed `method` writes a top-level `method:` line that
# `_build_config` never reads, and the run silently keeps the default method.
CONFIG_SECTION = "segmentation"


def _section_key(field: str) -> str:
    """Address a field inside the `segmentation` section of the config file."""
    return f"{CONFIG_SECTION}.{field}"


def _apply_overrides(data: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = {key: (dict(val) if isinstance(val, dict) else val) for key, val in data.items()}
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        node = merged
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = value
    return merged


def _build_config(data: Mapping[str, Any]) -> SegmentationConfig:
    segment = data.get(CONFIG_SECTION, data)
    if not isinstance(segment, Mapping):
        raise ValueError("config section 'segmentation' must be a mapping")
    consumed = {
        "method",
        "confidence_keep_ratio",
        "confidence_quantile_method",
        "corr_iou_thresh_intra",
        "corr_iou_thresh_inter",
        "depth_merge_thresh",
        "window_size",
        "overlap",
        "depth_refine",
        "sample_interval",
    }
    kwargs: dict[str, Any] = {
        key: segment[key] for key in consumed if key in segment
    }
    for nested, cls in (
        ("felzenszwalb", FelzenszwalbConfig),
        ("geometry", GeometryConfig),
        ("atomic", AtomicConfig),
        ("irls", IRLSConfig),
    ):
        if nested in segment:
            payload = segment[nested]
            if not isinstance(payload, Mapping):
                raise ValueError(f"config section 'segmentation.{nested}' must be a mapping")
            kwargs[nested] = cls(**dict(payload))
    unknown = set(segment) - consumed - {"felzenszwalb", "geometry", "atomic", "irls"}
    if unknown:
        raise ValueError(
            "unknown segmentation config keys: " + ", ".join(sorted(unknown))
        )
    return SegmentationConfig(**kwargs)


def load_segmentation_config(
    path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> SegmentationConfig:
    """Load the locked configuration, then apply explicit overrides.

    `method` is the one field a comparison is expected to override, via
    `LASER_SEGMENTATION_METHOD` or an explicit override. Every other override is recorded in the
    run identity so a reader can see exactly what differed.
    """
    import os

    resolved = path or os.environ.get(ENV_CONFIG_PATH) or DEFAULT_CONFIG_PATH
    config_path = Path(resolved)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"segmentation config not found: {config_path} "
            f"(set {ENV_CONFIG_PATH} or pass an explicit path)"
        )
    data = _load_yaml_mapping(config_path)
    env_method = os.environ.get(ENV_METHOD)
    if env_method:
        data = _apply_overrides(data, {_section_key("method"): env_method})
    if overrides:
        data = _apply_overrides(data, overrides)
    return _build_config(data)


def add_cli_arguments(parser, *, default_method: str | None = None) -> None:
    """Add the shared comparison arguments to an entry point's argument parser.

    Every entry point calls this, so a comparison is launched identically from `demo.py`,
    `demo_lc.py`, `eval_launch.py` and the `mv_recon` evaluators:

        python <entry>.py ... --segmentation_method geometry --segmentation_diagnostics

    `method` is the one intended independent variable. `--segmentation_config` exists because the
    locked file is the single parameter source; pointing it elsewhere makes a deliberate deviation
    explicit rather than implicit.
    """
    parser.add_argument(
        '--segmentation_config',
        default=None,
        type=str,
        help=f'locked segmentation config (default: {DEFAULT_CONFIG_PATH}, '
             f'or the {ENV_CONFIG_PATH} environment variable)',
    )
    parser.add_argument(
        '--segmentation_method',
        default=default_method,
        choices=list(SEGMENTATION_METHODS),
        help='initial segmentation method; the only intended independent variable',
    )
    parser.add_argument(
        '--segmentation_diagnostics',
        action='store_true',
        help='write per-window OP-1..OP-6 diagnostics beside the window caches',
    )
    parser.add_argument(
        '--segmentation_diagnostics_dir',
        default=None,
        type=str,
        help='optional explicit directory for diagnostics output',
    )
    parser.add_argument(
        '--segmentation_dump_arrays',
        action='store_true',
        help='also dump full-resolution label/mask arrays (implies observation level arrays)',
    )


def resolve_from_args(args) -> tuple["SegmentationConfig", "DiagnosticsConfig"]:
    """Resolve `(SegmentationConfig, DiagnosticsConfig)` from a parsed argument namespace.

    Only the fields the shared CLI exposes may differ from the locked file. `window_size`,
    `overlap`, `depth_refine` and every threshold come from the config, which is what makes runs
    from different entry points comparable at all.
    """
    overrides: dict[str, Any] = {}
    method = getattr(args, "segmentation_method", None)
    if method:
        overrides[_section_key("method")] = method
    config = load_segmentation_config(getattr(args, "segmentation_config", None), overrides)

    # `--segmentation_dump_arrays` alone must also enable observation. Otherwise it sets a level
    # that `DiagnosticsConfig.active` still reports as off, and a flag asking for the MOST output
    # would silently produce none.
    dump_arrays = bool(getattr(args, "segmentation_dump_arrays", False))
    enabled = bool(getattr(args, "segmentation_diagnostics", False)) or dump_arrays
    level = "arrays" if dump_arrays else ("scalar" if enabled else "off")
    diagnostics = DiagnosticsConfig(
        enabled=enabled,
        level=level,
        output_dir=getattr(args, "segmentation_diagnostics_dir", None),
        dump_arrays=dump_arrays,
    )
    return config, diagnostics


def _env_flag(name: str, default: bool = False) -> bool:
    import os

    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def resolve_from_environment() -> tuple["SegmentationConfig", "DiagnosticsConfig"]:
    """Resolve the pair without an argument parser.

    Needed by the Hydra-driven evaluators (`mv_recon/eval.py`, `mv_recon/eval_outdoor.py`), whose
    command line belongs to Hydra and cannot accept the shared comparison flags. Both spellings
    exist so that every entry point can be driven uniformly from a shell loop:

        LASER_SEGMENTATION_METHOD=geometry LASER_SEGMENTATION_DIAGNOSTICS=1 \\
            python mv_recon/eval.py
    """
    import os

    overrides: dict[str, Any] = {}
    method = os.environ.get(ENV_METHOD)
    if method:
        overrides[_section_key("method")] = method
    config = load_segmentation_config(None, overrides)

    enabled = _env_flag(ENV_DIAGNOSTICS)
    diagnostics = DiagnosticsConfig(
        enabled=enabled,
        level="scalar" if enabled else "off",
        output_dir=os.environ.get(ENV_DIAGNOSTICS_DIR) or None,
        dump_arrays=False,
    )
    return config, diagnostics


def describe_config(config: SegmentationConfig, diagnostics: DiagnosticsConfig | None = None) -> str:
    """Human-readable one-block summary, printed once per run for the record."""
    lines = ["segmentation config", "-------------------"]
    identity = config.run_identity()
    for key, value in identity.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                lines.append(f"  {key}.{sub_key}: {sub_value}")
        else:
            lines.append(f"  {key}: {value}")
    lines.append(f"  run_identity_hash: {config.run_identity_hash()}")
    if diagnostics is not None:
        lines.append("diagnostics config")
        lines.append("------------------")
        lines.append(f"  enabled: {diagnostics.enabled}")
        lines.append(f"  level: {diagnostics.level}")
        lines.append(f"  output_dir: {diagnostics.output_dir}")
    return "\n".join(lines)


__all__ = [
    "METHOD_DEPTH",
    "METHOD_GEOMETRY",
    "METHOD_ATOMIC",
    "SEGMENTATION_METHODS",
    "DEFAULT_CONFIG_PATH",
    "ENV_CONFIG_PATH",
    "ENV_METHOD",
    "FelzenszwalbConfig",
    "GeometryConfig",
    "AtomicConfig",
    "IRLSConfig",
    "SegmentationConfig",
    "DiagnosticsConfig",
    "load_segmentation_config",
    "add_cli_arguments",
    "resolve_from_args",
    "resolve_from_environment",
    "describe_config",
]
