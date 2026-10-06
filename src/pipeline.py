"""Pure, testable analysis orchestration independent of Tkinter."""
from dataclasses import dataclass
import numpy as np
from .analysis import subtract_baseline, smooth_signal, fit_peak, compute_momentum

@dataclass(frozen=True)
class PipelineConfig:
    baseline_enabled: bool = False
    baseline_x_min: float | None = None
    baseline_x_max: float | None = None
    baseline_degree: int = 1
    smoothing_enabled: bool = False
    smoothing_window: int = 11
    fit_enabled: bool = False
    fit_model: str = "gaussian"
    momentum_degree: int = 1
    momentum_threshold: float = 5.0

@dataclass
class AnalysisResult:
    x: np.ndarray
    y_raw: np.ndarray
    y_corrected: np.ndarray
    baseline: np.ndarray
    y_smoothed: np.ndarray | None
    y_fitted: np.ndarray | None
    fit_params: dict
    elution_x: float | None
    threshold_y: float

def run_pipeline(x, y, config: PipelineConfig = PipelineConfig()):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x and y must be one-dimensional arrays of equal length")
    if len(x) < 3:
        raise ValueError("At least three data points are required")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must contain only finite values")
    order = np.argsort(x)
    x, y = x[order], y[order]
    if config.baseline_enabled:
        corrected, baseline = subtract_baseline(x, y, config.baseline_x_min, config.baseline_x_max, config.baseline_degree)
    else:
        corrected, baseline = y.copy(), np.zeros_like(y)
    smoothed = smooth_signal(corrected, config.smoothing_window) if config.smoothing_enabled else None
    working = smoothed if smoothed is not None else corrected
    fitted, params = None, {}
    if config.fit_enabled:
        try:
            fitted, params = fit_peak(x, working, config.fit_model)
            working = fitted
        except Exception as exc:
            params = {"error": str(exc)}
    elution = compute_momentum(x, working, config.momentum_degree, config.momentum_threshold)
    ymax = float(np.max(np.abs(working))) if len(working) else 0.0
    threshold_y = config.momentum_threshold / 100.0 * ymax if ymax > 0 else 0.0
    return AnalysisResult(x, y, corrected, baseline, smoothed, fitted, params, elution, threshold_y)

def run_file_pipeline(fd):
    """Canonical adapter from the GUI's FileData object to the pure pipeline."""
    from .state import normalize_analysis_axes
    normalize_analysis_axes(fd)
    x_raw, y_raw = fd.get_xy()
    if len(x_raw) == 0:
        return {}
    cfg = PipelineConfig(
        baseline_enabled=fd.baseline_enabled,
        baseline_x_min=fd.baseline_x_min,
        baseline_x_max=fd.baseline_x_max,
        baseline_degree=fd.baseline_degree,
        smoothing_enabled=fd.smoothing_enabled,
        smoothing_window=fd.smoothing_window,
        fit_enabled=fd.fit_enabled,
        fit_model=fd.fit_model,
        momentum_degree=fd.momentum_degree,
        momentum_threshold=fd.momentum_threshold,
    )
    result = run_pipeline(x_raw, y_raw, cfg)
    return {
        "x_raw": x_raw, "y_raw": y_raw, "x": result.x,
        "y": result.y_corrected, "y_raw_for_baseline": result.y_raw,
        "baseline_curve": result.baseline, "y_smooth": result.y_smoothed,
        "y_fit": result.y_fitted, "fit_params": result.fit_params,
        "elution_x": result.elution_x, "threshold_y": result.threshold_y,
    }
