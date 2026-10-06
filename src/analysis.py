"""
analysis.py
===========
Chromatographic peak analysis:
  - Baseline correction (polynomial fit on specified x-ranges)
  - Signal smoothing (Savitzky-Golay or uniform moving average)
  - Peak fitting (Gaussian, Double-Gaussian, Exponentially Modified Gaussian)
  - Elution volume calculation via statistical moment (momentum of degree n)
"""

from __future__ import annotations
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
from typing import Optional


# ---------------------------------------------------------------------------
# Baseline correction
# ---------------------------------------------------------------------------

def fit_baseline(
    x: np.ndarray,
    y: np.ndarray,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    degree: int = 1,
) -> np.ndarray:
    """
    Fit a polynomial baseline to the regions outside the peak window
    [x_min, x_max] and return the baseline evaluated at all x points.

    If x_min / x_max are None, baseline region is the lower 20% and
    upper 20% of the x range.
    """
    if len(x) == 0:
        return np.zeros_like(y)

    if x_min is None:
        x_min = x[0] + 0.2 * (x[-1] - x[0])
    if x_max is None:
        x_max = x[-1] - 0.2 * (x[-1] - x[0])

    # Use regions outside [x_min, x_max] for the polynomial fit
    mask = (x < x_min) | (x > x_max)
    if np.sum(mask) < degree + 1:
        # Not enough points outside; use all points
        mask = np.ones_like(x, dtype=bool)

    coeffs = np.polyfit(x[mask], y[mask], degree)
    return np.polyval(coeffs, x)


def subtract_baseline(
    x: np.ndarray,
    y: np.ndarray,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    degree: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_corrected, baseline)."""
    baseline = fit_baseline(x, y, x_min, x_max, degree)
    return y - baseline, baseline


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def smooth_signal(
    y: np.ndarray,
    window: int = 11,
    method: str = "savgol",
    polyorder: int = 3,
) -> np.ndarray:
    """
    Smooth the signal.

    Parameters
    ----------
    window   : window length (must be odd for Savitzky-Golay)
    method   : 'savgol' | 'uniform'
    polyorder: polynomial order for Savitzky-Golay
    """
    window = max(3, window)
    if window % 2 == 0:
        window += 1  # ensure odd for savgol

    n = len(y)
    if n < window:
        return y.copy()

    if method == "savgol":
        po = min(polyorder, window - 1)
        return savgol_filter(y, window_length=window, polyorder=po)
    else:  # uniform
        return uniform_filter1d(y.astype(float), size=window)


# ---------------------------------------------------------------------------
# Peak fitting models
# ---------------------------------------------------------------------------

def _gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _split_gaussian(x, A, mu, sigma_L, sigma_R):
    """
    Split (bifurcated) Gaussian: one amplitude and centre, but independent
    half-widths on the left (sigma_L) and right (sigma_R) sides.
    """
    y = np.where(
        x < mu,
        A * np.exp(-0.5 * ((x - mu) / sigma_L) ** 2),
        A * np.exp(-0.5 * ((x - mu) / sigma_R) ** 2),
    )
    return y


def _emg(x, A, mu, sigma, lam):
    """Exponentially Modified Gaussian (EMG)."""
    from scipy.special import log_ndtr
    lam = max(lam, 1e-9)
    sigma = max(sigma, np.finfo(float).tiny)
    z = (mu - x) / (np.sqrt(2) * sigma) + sigma * lam / np.sqrt(2)
    # Evaluate erfc and the exponential together in log space.  During a
    # fit, the two factors can overflow and underflow separately even when
    # their product is finite.
    log_value = (
        np.log(lam / 2)
        + 0.5 * (lam * sigma) ** 2
        - lam * (x - mu)
        + np.log(2) + log_ndtr(-z * np.sqrt(2))
    )
    limits = np.log(np.finfo(float).tiny), np.log(np.finfo(float).max)
    return A * np.exp(np.clip(log_value, *limits))


_FIT_MODELS = {
    "gaussian":       (_gaussian,       ["A", "mu", "sigma"]),
    "split_gaussian": (_split_gaussian, ["A", "mu", "sigma_L", "sigma_R"]),
    "emg":            (_emg,            ["A", "mu", "sigma", "lambda"]),
}

FIT_MODEL_NAMES = list(_FIT_MODELS.keys())


def fit_peak(
    x: np.ndarray,
    y: np.ndarray,
    model: str = "gaussian",
) -> tuple[np.ndarray, dict]:
    """
    Fit a peak model to (x, y).

    Returns
    -------
    y_fit    : fitted curve evaluated at x
    params   : dict of fitted parameter names → values
    """
    if model not in _FIT_MODELS:
        raise ValueError(f"Unknown model: {model!r}. Choose from {FIT_MODEL_NAMES}")

    fn, param_names = _FIT_MODELS[model]

    # Build initial parameter guesses
    peak_idx = np.argmax(np.abs(y))
    A0 = y[peak_idx]
    mu0 = x[peak_idx]
    half = np.max(np.abs(y)) / 2
    above = x[np.abs(y) >= half]
    sigma0 = (above[-1] - above[0]) / 2.355 if len(above) >= 2 else (x[-1] - x[0]) / 10

    if model == "gaussian":
        p0 = [A0, mu0, sigma0]
        bounds = ([-np.inf, x[0], 0], [np.inf, x[-1], x[-1] - x[0]])
    elif model == "split_gaussian":
        p0 = [A0, mu0, sigma0, sigma0]
        bounds = ([-np.inf, x[0], 0, 0], [np.inf, x[-1], x[-1] - x[0], x[-1] - x[0]])
    elif model == "emg":
        p0 = [A0, mu0, sigma0, 1.0]
        bounds = ([-np.inf, x[0], 0, 0], [np.inf, x[-1], x[-1] - x[0], np.inf])
    else:
        p0 = [1.0] * len(param_names)
        bounds = (-np.inf, np.inf)

    try:
        popt, _ = curve_fit(fn, x, y, p0=p0, bounds=bounds, maxfev=10000)
    except RuntimeError:
        popt = p0

    y_fit = fn(x, *popt)
    params = dict(zip(param_names, popt))
    return y_fit, params


# ---------------------------------------------------------------------------
# Elution volume via statistical moment (momentum of degree n)
# ---------------------------------------------------------------------------

def compute_momentum(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 1,
    threshold_pct: float = 5.0,
) -> Optional[float]:
    """
    Calculate the elution position as the statistical moment of degree `degree`
    of the signal above `threshold_pct` percent of the peak height.

    Parameters
    ----------
    x            : x-axis array (volume / time)
    y            : baseline-corrected signal
    degree       : moment degree (1 = centroid, higher = more sensitive to shape)
    threshold_pct: percentage of the peak maximum to use as the lower cutoff

    Returns
    -------
    x_elution : elution position (same units as x), or None if computation fails
    """
    if len(x) < 3:
        return None

    y_abs = np.abs(y)
    y_max = np.max(y_abs)
    if y_max == 0:
        return None

    threshold = threshold_pct / 100.0 * y_max
    mask = y_abs >= threshold
    if np.sum(mask) < 2:
        return None

    xm = x[mask]
    ym = y_abs[mask]

    # Statistical moment: sum(y * x^degree) / sum(y * x^(degree-1))
    # For degree=1 this is the centroid sum(y*x)/sum(y)
    if degree == 1:
        elution = np.sum(ym * xm) / np.sum(ym)
    else:
        # Generalised moment: E[X^degree] / E[X^(degree-1)]
        elution = np.sum(ym * xm ** degree) / np.sum(ym * xm ** (degree - 1))

    return float(elution)


# ---------------------------------------------------------------------------
# Full analysis pipeline for a single FileData
# ---------------------------------------------------------------------------

def analyze_file(fd) -> dict:
    """Backward-compatible entry point delegating to the canonical pipeline."""
    from .pipeline import run_file_pipeline
    return run_file_pipeline(fd)
