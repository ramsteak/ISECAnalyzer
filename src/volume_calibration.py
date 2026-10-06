"""Experiment-wide volume calibration for ISEC datasets."""
from dataclasses import dataclass
import numpy as np
from scipy.stats import linregress
from .data_io import _TO_G, _TO_MIN

@dataclass
class CalibrationResult:
    mode: str
    flow_rate_ml_min: float | None = None
    density_g_ml: float | None = None
    mass_flow_g_min: float | None = None
    slope: float | None = None
    intercept: float | None = None
    intercepts: list[float] | None = None
    r2: float | None = None
    n_points: int = 0

def _clean(x, y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()<2: raise ValueError("At least two finite calibration points are required.")
    return x[ok],y[ok]

def regress_mass_flow(time_values, time_unit, weight_values, weight_unit):
    t=np.asarray(time_values,float)*_TO_MIN.get(time_unit,1.0)
    m=np.asarray(weight_values,float)*_TO_G.get(weight_unit,1.0)
    t,m=_clean(t,m)
    fit=linregress(t,m)
    return CalibrationResult("regression", mass_flow_g_min=float(fit.slope), slope=float(fit.slope),
        intercept=float(fit.intercept), r2=float(fit.rvalue**2), n_points=len(t))

def volume_from_time(time_values, unit, flow_rate):
    return np.asarray(time_values,float)*_TO_MIN.get(unit,1.0)*float(flow_rate)

def volume_from_weight(weight_values, unit, density):
    return np.asarray(weight_values,float)*_TO_G.get(unit,1.0)/float(density)


def regress_mass_flow_per_file(time_weight_pairs, time_unit, weight_unit):
    """Fit one common slope with an independent intercept for each file."""
    groups = []
    for t, m in time_weight_pairs:
        t = np.asarray(t, float) * _TO_MIN.get(time_unit, 1.0)
        m = np.asarray(m, float) * _TO_G.get(weight_unit, 1.0)
        ok = np.isfinite(t) & np.isfinite(m)
        if ok.sum() >= 2:
            groups.append((t[ok], m[ok]))
    if not groups:
        raise ValueError("At least one file with two finite time/weight points is required.")
    den = sum(np.sum((t - t.mean()) ** 2) for t, m in groups)
    if den <= 0:
        raise ValueError("The calibration time values must vary within the files.")
    slope = sum(np.sum((t - t.mean()) * (m - m.mean())) for t, m in groups) / den
    intercepts = [float(m.mean() - slope * t.mean()) for t, m in groups]
    obs = np.concatenate([m for t, m in groups])
    pred = np.concatenate([slope * t + b for (t, m), b in zip(groups, intercepts)])
    ss_res = float(np.sum((obs - pred) ** 2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return CalibrationResult("regression", mass_flow_g_min=float(slope), slope=float(slope),
        intercept=float(np.mean(intercepts)), intercepts=intercepts, r2=float(r2),
        n_points=int(len(obs)))
