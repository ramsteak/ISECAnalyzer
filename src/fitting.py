"""
fitting.py
==========
ISEC pore-model fitting.

Models supported:
  - Cylindrical pore model (Jerabek)  – K_ij = (1 - Rh_i / r_j)^2
  - Ogston network model               – K_ij = exp(-pi/4 * c_j * (Rh_i + r_chain)^2)
  - Knox-Scott rational-function fit  – ratfun(Rh, p, q) + pore-size distribution

The fitting solves the non-negative least squares problem:
    Ve_i = V0 + sum_j( K_ij * v_j )
where v_j are the pore volumes to be optimised (non-negative).

Improvements over ISECrunch76:
  - Uses scipy.optimize.nnls as a fast warm-start, then lmfit for uncertainty estimation
  - Two-pass strategy: fix near-zero pore fractions in the second pass
  - Returns rich result object with covariances, residuals, PSD
"""

from __future__ import annotations
import numpy as np
import numpy.linalg as nla
from scipy.optimize import nnls, curve_fit
from scipy.stats import t
import lmfit
from typing import Optional, NamedTuple


# ---------------------------------------------------------------------------
# Partition coefficient matrices
# ---------------------------------------------------------------------------

def K_cylindrical(Rh: np.ndarray, pore_radii: np.ndarray) -> np.ndarray:
    """
    Partition coefficient for cylindrical pore model.

    K[i,j] = max(0, (1 - Rh[i] / r[j]))^2

    Parameters
    ----------
    Rh         : (n,) array of hydrodynamic radii [nm]
    pore_radii : (m,) array of pore radii [nm]

    Returns
    -------
    K : (n, m) matrix
    """
    Rh = np.asarray(Rh, dtype=float)[:, None]     # (n,1)
    r  = np.asarray(pore_radii, dtype=float)[None, :]  # (1,m)
    ratio = Rh / r
    return np.where(ratio < 1, (1 - ratio) ** 2, 0.0)


def K_ogston(Rh: np.ndarray, chain_conc: np.ndarray, chain_diam: float = 0.4) -> np.ndarray:
    """
    Partition coefficient for Ogston (fibre network) model.

    K[i,j] = exp(-pi/4 * c[j] * (Rh[i] + r_chain)^2)

    Parameters
    ----------
    Rh         : (n,) hydrodynamic radii [nm]
    chain_conc : (m,) fibre concentrations [nm/nm^3]
    chain_diam : fibre diameter [nm]  (default 0.4)

    Returns
    -------
    K : (n, m) matrix
    """
    Rh = np.asarray(Rh, dtype=float)[:, None]
    c  = np.asarray(chain_conc, dtype=float)[None, :]
    r_chain = chain_diam / 2.0
    return np.exp(-0.25 * np.pi * c * (Rh + r_chain) ** 2)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

class FitResult(NamedTuple):
    pore_params: np.ndarray     # pore radii [nm] or chain concentrations
    pore_volumes: np.ndarray    # optimised pore volumes [ml]
    spec_volumes: np.ndarray    # specific pore volumes [ml/g]  (NaN if weight unknown)
    std_errors: np.ndarray      # standard errors on pore volumes
    spec_surface: np.ndarray    # specific surface [m^2/g] (cyl only)
    Ve_fit: np.ndarray          # fitted elution volumes
    Ve_obs: np.ndarray          # observed elution volumes
    Rh: np.ndarray              # hydrodynamic radii used
    ssr: float                  # sum of squared residuals
    message: str
    n_iter: int
    residual_volume: float      # Vc - V0 - sum(v)
    model: str                  # 'cylindrical' | 'ogston'
    weight: float               # sample weight [g]  (1 if unknown)
    V0: float
    Vc: float
    r2: float
    converged: bool
    n_data: int
    n_params: int
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    ve_ci_lower: np.ndarray
    ve_ci_upper: np.ndarray
    confidence_level: float


# ---------------------------------------------------------------------------
# Core fitter
# ---------------------------------------------------------------------------

class ISECFitter:
    """
    Fits pore volume distribution from ISEC elution data.

    Usage
    -----
    fitter = ISECFitter(
        Rh=[0.3, 1.5, 5.0, 15.0, 30.0],
        Ve=[3.1, 2.8, 2.2, 1.5, 0.8],
        V0=0.5,
        Vc=4.0,
        model='cylindrical',
        pore_params=[2, 5, 10, 20, 40, 80],
    )
    result = fitter.fit()
    """

    def __init__(
        self,
        Rh: list[float],
        Ve: list[float],
        V0: float,
        Vc: float,
        model: str = "cylindrical",
        pore_params: Optional[list[float]] = None,
        chain_diam: float = 0.4,
        weight: float = 1.0,
        fit_v0: bool = False,
    ):
        self.Rh = np.asarray(Rh, dtype=float)
        self.Ve_obs = np.asarray(Ve, dtype=float)
        self.V0 = float(V0)
        self.Vc = float(Vc)
        self.model = model.lower()
        self.chain_diam = chain_diam
        self.weight = weight if weight > 0 else 1.0
        self.fit_v0 = fit_v0

        # Default pore parameter sets
        if pore_params is None:
            if self.model == "cylindrical":
                pore_params = [2, 5, 10, 15, 20, 30, 40, 60, 80, 100, 150]
            else:
                pore_params = [0.05, 0.1, 0.2, 0.4, 0.8, 1.5]
        self.pore_params = np.asarray(pore_params, dtype=float)

        # Compute K matrix: K @ v = Ve_net (net elution volume)
        self.Ve_net = self.Ve_obs - self.V0
        self._build_K()

    def _build_K(self):
        if self.model == "cylindrical":
            self.K = K_cylindrical(self.Rh, self.pore_params)
        elif self.model == "ogston":
            self.K = K_ogston(self.Rh, self.pore_params, self.chain_diam)
        else:
            raise ValueError(f"Unknown model: {self.model!r}")

    def fit(self) -> FitResult:
        """Run the two-pass non-negative least squares fit, optionally fitting V0."""
        # --- Pass 1: fast NNLS warm start (fixed V0) ---
        v0_nnls, ssr0 = nnls(self.K, self.Ve_net)

        # --- Build lmfit parameters (all non-negative) ---
        params = lmfit.Parameters()
        for i, pp in enumerate(self.pore_params):
            name = f"v{i:03d}"
            params.add(name, value=max(v0_nnls[i], 1e-6), min=0.0)

        # Optionally fit V0 as a free parameter (constrained to be <= min(Ve))
        if self.fit_v0:
            v0_max = float(np.min(self.Ve_obs)) if len(self.Ve_obs) > 0 else self.V0
            params.add("V0", value=self.V0, min=0.0, max=v0_max)

        def residuals(p):
            v = np.array([p[f"v{i:03d}"].value for i in range(len(self.pore_params))])
            v0_val = p["V0"].value if "V0" in p else self.V0
            ve_net = self.Ve_obs - v0_val
            return np.dot(self.K, v) - ve_net

        # leastsq requires M (data points) >= N (parameters); use Nelder-Mead otherwise
        method = "leastsq" if len(self.Ve_net) >= len(self.pore_params) else "nelder"
        res1 = lmfit.minimize(residuals, params, method=method)
        n_iter = res1.nfev

        # --- Pass 2: fix near-zero pore fractions ---
        zero_thresh = 1e-4
        v1 = np.array([res1.params[f"v{i:03d}"].value for i in range(len(self.pore_params))])
        params2 = lmfit.Parameters()
        for i, pp in enumerate(self.pore_params):
            name = f"v{i:03d}"
            if v1[i] < zero_thresh:
                params2.add(name, value=0.0, vary=False)
            else:
                params2.add(name, value=v1[i], min=0.0)
        # Carry over V0 fit parameter to pass 2
        if self.fit_v0 and "V0" in res1.params:
            v0_val = res1.params["V0"].value
            v0_max = float(np.min(self.Ve_obs)) if len(self.Ve_obs) > 0 else self.V0
            params2.add("V0", value=v0_val, min=0.0, max=v0_max)

        n_active = sum(1 for i in range(len(self.pore_params))
                       if params2[f"v{i:03d}"].vary)
        method2 = "leastsq" if len(self.Ve_net) >= max(n_active, 1) else "nelder"
        res2 = lmfit.minimize(residuals, params2, method=method2)
        n_iter += res2.nfev

        # --- Extract results ---
        v_opt = np.array([res2.params[f"v{i:03d}"].value for i in range(len(self.pore_params))])
        try:
            stderr = np.array([
                res2.params[f"v{i:03d}"].stderr or 0.0
                for i in range(len(self.pore_params))
            ])
        except Exception:
            stderr = np.zeros_like(v_opt)

        # If V0 was fitted, update self.V0 with the optimised value
        if self.fit_v0 and "V0" in res2.params:
            self.V0 = float(res2.params["V0"].value)

        Ve_fit = np.dot(self.K, v_opt) + self.V0
        ssr = float(np.sum((Ve_fit - self.Ve_obs) ** 2))
        residual_vol = self.Vc - self.V0 - np.sum(v_opt)

        spec_vol = v_opt / self.weight
        spec_err = stderr / self.weight

        if self.model == "cylindrical":
            # Specific surface [m^2/g]: 4 * V_spec / d_pore  (d_pore in nm → *1e-9 m)
            # = 4 * spec_vol_cm3/g * 1e-6 m3/cm3 / (pore_diam_nm * 1e-9 m)
            # = spec_vol * 4e-6 / (pore_diam * 1e-9) = spec_vol * 4000 / pore_diam  [m2/g]
            with np.errstate(divide="ignore", invalid="ignore"):
                spec_surf = np.where(
                    self.pore_params > 0,
                    4000.0 * spec_vol / self.pore_params,
                    0.0,
                )
        else:
            # Ogston: amount of polymer [1e10 m/g] ∝ conc * spec_vol
            spec_surf = 100.0 * self.pore_params * spec_vol

        # Approximate two-sided 95% confidence intervals using the covariance
        # matrix returned by LMFit.  Important: ``result.covar`` contains
        # only the varying parameters, in ``result.var_names`` order.  It
        # must therefore not be indexed as if it contained every parameter,
        # because fixed pore-volume parameters may be present.
        confidence_level = 0.95
        dof = max(int(len(self.Ve_obs) - getattr(res2, "nvarys", 0)), 1)
        tcrit = float(t.ppf(0.5 + confidence_level / 2.0, dof))

        n_pores = len(self.pore_params)
        ci_se = np.full(n_pores, np.nan, dtype=float)
        cov_v = np.zeros((n_pores, n_pores), dtype=float)
        has_covariance = False
        cov = getattr(res2, "covar", None)
        var_names = list(getattr(res2, "var_names", []) or [])

        if cov is not None and var_names:
            cov = np.asarray(cov, dtype=float)
            if (cov.ndim == 2 and cov.shape[0] == len(var_names)
                    and cov.shape[1] == len(var_names)
                    and np.all(np.isfinite(cov))):
                pore_indices = {}
                for i in range(n_pores):
                    name = f"v{i:03d}"
                    if name in var_names:
                        pore_indices[i] = var_names.index(name)

                # Map the varying-parameter covariance into the complete
                # pore-volume parameter space. Fixed parameters retain zero
                # variance and zero covariance.
                for i, ii in pore_indices.items():
                    for j, jj in pore_indices.items():
                        cov_v[i, j] = cov[ii, jj]
                    ci_se[i] = np.sqrt(max(cov_v[i, i], 0.0))
                has_covariance = bool(pore_indices)

        ci_half = tcrit * ci_se
        ci_lower = np.where(np.isfinite(ci_half),
                            np.maximum(0.0, v_opt - ci_half), np.nan)
        ci_upper = np.where(np.isfinite(ci_half),
                            np.maximum(0.0, v_opt + ci_half), np.nan)

        # Propagate the covariance to the fitted exclusion-volume curve.
        # Only the fitted pore-volume parameters contribute here.  If no
        # usable covariance is available, retain NaN bands rather than
        # reporting misleading zero-width intervals.
        rang_ci = np.geomspace(
            max(0.05, 0.9 * self.Rh.min()), 1.1 * self.Rh.max(), 500
        )
        if has_covariance:
            try:
                K_curve = (K_cylindrical(rang_ci, self.pore_params)
                           if self.model == "cylindrical" else
                           K_ogston(rang_ci, self.pore_params, self.chain_diam))
                curve_var = np.einsum("ij,jk,ik->i", K_curve, cov_v, K_curve)
                curve_half = tcrit * np.sqrt(np.maximum(curve_var, 0.0))
                ve_curve = np.dot(K_curve, v_opt) + self.V0
                ve_ci_lower = ve_curve - curve_half
                ve_ci_upper = ve_curve + curve_half
            except Exception:
                ve_ci_lower = np.full(len(rang_ci), np.nan)
                ve_ci_upper = np.full(len(rang_ci), np.nan)
        else:
            ve_ci_lower = np.full(len(rang_ci), np.nan)
            ve_ci_upper = np.full(len(rang_ci), np.nan)

        return FitResult(
            pore_params=self.pore_params,
            pore_volumes=v_opt,
            spec_volumes=spec_vol,
            std_errors=spec_err,
            spec_surface=spec_surf,
            Ve_fit=Ve_fit,
            Ve_obs=self.Ve_obs,
            Rh=self.Rh,
            ssr=ssr,
            message=res2.message,
            n_iter=n_iter,
            residual_volume=float(residual_vol),
            model=self.model,
            weight=self.weight,
            V0=self.V0,
            Vc=self.Vc,
            r2=float(1.0 - ssr / np.sum((self.Ve_obs - np.mean(self.Ve_obs)) ** 2)) if np.sum((self.Ve_obs - np.mean(self.Ve_obs)) ** 2) > 0 else 1.0,
            converged=bool(getattr(res2, "success", True)),
            n_data=int(len(self.Ve_obs)),
            n_params=int(len(self.pore_params) + (1 if self.fit_v0 else 0)),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ve_ci_lower=ve_ci_lower,
            ve_ci_upper=ve_ci_upper,
            confidence_level=confidence_level,
        )


# ---------------------------------------------------------------------------
# Knox-Scott rational function fit
# ---------------------------------------------------------------------------

def _rat(x: np.ndarray, p: list, q: list) -> np.ndarray:
    """Rational function P(x) / Q(x) where Q has implicit leading 1."""
    return np.polyval(p, x) / np.polyval(q + [1.0], x)


def _rat_fn(x, nnu, nde, *args):
    p = list(args[:nnu])
    q = list(args[nnu:nnu + nde])
    return _rat(x, p, q)


def _polyder_val(P, n, x):
    return np.polyval(np.polyder(np.poly1d(P), m=n), x)


def _psd_from_rat(x: np.ndarray, nnu: int, nde: int, *args) -> np.ndarray:
    """
    Knox-Scott pore-size distribution from rational function parameters.
    Returns dV/dr evaluated at x.
    """
    p = list(args[:nnu])
    q = list(args[nnu:nnu + nde])
    Px = np.poly1d(p)
    Qx = np.poly1d(q + [1.0])

    P0 = np.polyval(Px, x);  P1 = _polyder_val(Px, 1, x)
    P2 = _polyder_val(Px, 2, x);  P3 = _polyder_val(Px, 3, x)
    Q0 = np.polyval(Qx, x);  Q1 = _polyder_val(Qx, 1, x)
    Q2 = _polyder_val(Qx, 2, x);  Q3 = _polyder_val(Qx, 3, x)

    R0 =  1 / Q0
    R1 = -Q1 / Q0 ** 2
    R2 = (2 * Q1 ** 2 - Q0 * Q2) / Q0 ** 3
    R3 = (6 * Q0 * Q1 * Q2 - Q3 * Q0 ** 2 - 6 * Q1 ** 3) / Q0 ** 4

    return -x ** 2 * (P3 * R0 + 3 * P2 * R1 + 3 * P1 * R2 + P0 * R3) / 2


class KnoxScottResult(NamedTuple):
    Rh: np.ndarray
    Ve_obs: np.ndarray
    Ve_fit: np.ndarray
    rang: np.ndarray
    Ve_curve: np.ndarray
    psd: np.ndarray         # dV/dr curve
    popt: np.ndarray
    nnu: int
    nde: int
    ssr: float
    weight: float
    r2: float
    converged: bool
    n_iter: int
    n_data: int
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    ve_ci_lower: np.ndarray
    ve_ci_upper: np.ndarray
    confidence_level: float


def fit_knox_scott(
    Rh: list[float],
    Ve: list[float],
    weight: float = 1.0,
    nnu: int = 1,
    nde: int = 3,
) -> KnoxScottResult:
    """
    Fit the Knox-Scott rational function to (Rh, Ve) data.

    Parameters
    ----------
    Rh     : hydrodynamic radii [nm]
    Ve     : elution volumes [ml]
    weight : sample weight [g]
    nnu    : numerator polynomial degree
    nde    : denominator polynomial degree
    """
    Rh = np.asarray(Rh, dtype=float)
    Ve = np.asarray(Ve, dtype=float)

    p0 = np.ones(nnu + nde)
    try:
        popt, pcov = curve_fit(
            lambda x, *p: _rat_fn(x, nnu, nde, *p),
            Rh, Ve, p0=p0,
            bounds=(0, np.inf),
            maxfev=50000,
        )
    except (RuntimeError, ValueError):
        popt = p0
        pcov = None

    rang = np.geomspace(max(0.05, 0.9 * Rh.min()), 1.1 * Rh.max(), 500)
    Ve_curve = _rat_fn(rang, nnu, nde, *popt)
    psd = _psd_from_rat(rang, nnu, nde, *popt) / weight

    Ve_fit = _rat_fn(Rh, nnu, nde, *popt)
    ssr = float(np.sum((Ve_fit - Ve) ** 2))
    tss = float(np.sum((Ve - Ve.mean()) ** 2))
    r2 = 1.0 - ssr / tss if tss > 0 else 1.0

    dof = max(int(len(Ve) - len(popt)), 1)
    tcrit = float(t.ppf(0.975, dof))
    if pcov is not None and np.all(np.isfinite(pcov)):
        se = np.sqrt(np.maximum(np.diag(pcov), 0.0))
        half = tcrit * se
        ci_lower = popt - half
        ci_upper = popt + half
        J = np.column_stack([
            (_rat_fn(rang, nnu, nde, *(popt + np.eye(len(popt))[j] * 1e-6)) -
             _rat_fn(rang, nnu, nde, *popt)) / 1e-6 for j in range(len(popt))
        ])
        var_curve = np.einsum("ij,jk,ik->i", J, pcov, J)
        half_curve = tcrit * np.sqrt(np.maximum(var_curve, 0.0))
        ve_ci_lower = Ve_curve - half_curve
        ve_ci_upper = Ve_curve + half_curve
    else:
        ci_lower = np.full(len(popt), np.nan)
        ci_upper = np.full(len(popt), np.nan)
        ve_ci_lower = np.full(len(rang), np.nan)
        ve_ci_upper = np.full(len(rang), np.nan)

    return KnoxScottResult(
        Rh=Rh, Ve_obs=Ve, Ve_fit=Ve_fit,
        rang=rang, Ve_curve=Ve_curve, psd=psd,
        popt=popt, nnu=nnu, nde=nde,
        ssr=ssr, weight=weight, r2=r2,
        converged=True, n_iter=0, n_data=int(len(Rh)),
        ci_lower=ci_lower, ci_upper=ci_upper,
        ve_ci_lower=ve_ci_lower, ve_ci_upper=ve_ci_upper,
        confidence_level=0.95,
    )
