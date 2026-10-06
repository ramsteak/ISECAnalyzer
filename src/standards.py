"""
standards.py
============
Hydrodynamic radius (Rh, in nm) calculations for all ISEC standard types.

Reference formulas (from ISECrunch / literature):
  Polystyrene (PS) :  Rh = 0.0246 * MW^0.588          (MW in g/mol, Rh in nm)
  Alkane (CnH2n+2) :  Rh = 0.5 * (n*0.45 + 3.32) / 10  (end-to-end / 10)
                       i.e. length [Å] = n*0.45 + 3.32 → Rh [nm] = length/10 * 0.1
                       (from original: diam = 0.1*(n*0.45+3.32), which is diameter)
  Dextran (D)      :  Rh = 0.0732 * MW^0.461           (MW in g/mol)
  Polymethacrylate :  Rh = 0.02346 * MW^0.568           (approximate; adjust as needed)
  Poly(ethylene glycol) / PEG: Rh = 0.02097 * MW^0.587
  D2O              :  Rh = 0.15  nm  (kinetic diameter of water ~0.3 nm → Rh ≈ 0.15)
  Ribose           :  Rh = 0.285 nm
  Xylose           :  Rh = 0.39 nm
  Sucrose          :  Rh = 0.675 nm
  Raffinose        :  Rh = 0.80 nm

Note: The original code uses "diam" which is actually the hydrodynamic radius Rh
(i.e. not the full diameter). We keep the same convention here.
"""

from __future__ import annotations
import re
import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# Registry: each entry is  (display_name, needs_MW, rh_function_or_constant)
# ---------------------------------------------------------------------------

_STANDARDS: dict[str, dict] = {
    "ps": {
        "label": "Polystyrene (PS)",
        "needs_mw": True,
        "unit": "Da",
        "rh_fn": lambda mw: 0.0246 * (mw ** 0.588),
        "abbrevs": ["ps", "polystyrene"],
    },
    "alkane": {
        "label": "Alkane (CnH2n+2)",
        "needs_mw": True,
        "unit": "#C",
        "rh_fn": lambda n: 0.1 * (n * 0.45 + 3.32),   # diam in nm (= Rh convention)
        "abbrevs": ["c", "alkane", "alk", "paraffin"],
    },
    "dextran": {
        "label": "Dextran",
        "needs_mw": True,
        "unit": "Da",
        "rh_fn": lambda mw: 0.0732 * (mw ** 0.461),
        "abbrevs": ["d", "dex", "dextran"],
    },
    "polymethacrylate": {
        "label": "Poly(methyl methacrylate) (PMMA)",
        "needs_mw": True,
        "unit": "Da",
        "rh_fn": lambda mw: 0.02346 * (mw ** 0.568),
        "abbrevs": ["p", "pmma", "polymethacrylate", "polymethylmethacrylate"],
    },
    "peg": {
        "label": "Poly(ethylene glycol) (PEG)",
        "needs_mw": True,
        "unit": "Da",
        "rh_fn": lambda mw: 0.02097 * (mw ** 0.587),
        "abbrevs": ["peg", "peo", "polyethyleneglycol"],
    },
    "d2o": {
        "label": "D2O (heavy water)",
        "needs_mw": False,
        "unit": None,
        "rh_fn": lambda _: 0.15,
        "abbrevs": ["d2o", "deuterium", "heavywater"],
    },
    "ribose": {
        "label": "Ribose",
        "needs_mw": False,
        "unit": None,
        "rh_fn": lambda _: 0.285,
        "abbrevs": ["rib", "ribose"],
    },
    "xylose": {
        "label": "Xylose",
        "needs_mw": False,
        "unit": None,
        "rh_fn": lambda _: 0.39,
        "abbrevs": ["xyl", "xylose"],
    },
    "sucrose": {
        "label": "Sucrose",
        "needs_mw": False,
        "unit": None,
        "rh_fn": lambda _: 0.675,
        "abbrevs": ["sac", "sacc", "sucrose"],
    },
    "raffinose": {
        "label": "Raffinose",
        "needs_mw": False,
        "unit": None,
        "rh_fn": lambda _: 0.80,
        "abbrevs": ["raff", "raffinose"],
    },
    "toluene": {
        "label": "Toluene",
        "needs_mw": False,
        "unit": None,
        "rh_fn": lambda _: 0.29,
        "abbrevs": ["tol", "toluene"],
    },
    "custom": {
        "label": "Custom (manual Rh)",
        "needs_mw": False,
        "unit": "nm",
        "rh_fn": lambda rh: rh,   # user supplies Rh directly
        "abbrevs": ["custom"],
    },
}

# Ordered list for comboboxes
STANDARD_KEYS = list(_STANDARDS.keys())
STANDARD_LABELS = [v["label"] for v in _STANDARDS.values()]


def get_standard_info(key: str) -> dict:
    return _STANDARDS.get(key, _STANDARDS["ps"])


def compute_rh(key: str, value: float) -> float:
    """Compute hydrodynamic radius (nm) for a given standard and its MW/size value."""
    info = _STANDARDS.get(key)
    if info is None:
        raise ValueError(f"Unknown standard key: {key!r}")
    return float(info["rh_fn"](value))


def standard_needs_mw(key: str) -> bool:
    return _STANDARDS.get(key, {}).get("needs_mw", True)


def standard_unit(key: str) -> Optional[str]:
    return _STANDARDS.get(key, {}).get("unit")


# ---------------------------------------------------------------------------
# Auto-detection from filename
# ---------------------------------------------------------------------------

_DETECT_PATTERNS = [
    (r"ps(\d+(?:[.]\d+)?)(k|m|million)?", "ps", lambda m: float(m.group(1)) * ({"k":1000,"m":1_000_000,"million":1_000_000}.get((m.group(2) or "").lower(), 1))),
    (r"polystyrene(\d+(?:[.]\d+)?)(k?)", "ps", lambda m: float(m.group(1)) * (1000 if m.group(2) == "k" else 1)),
    (r"(?<![a-z0-9])c\s*(\d{1,3})(?!\d)", "alkane", lambda m: float(m.group(1))),
    (r"(?:d|dextran)(\d+(?:[.]\d+)?)(k|m|million)?", "dextran", lambda m: float(m.group(1)) * ({"k":1000,"m":1_000_000,"million":1_000_000}.get((m.group(2) or "").lower(), 1))),
    (r"(?:p|pmma)(\d+(?:[.]\d+)?)(k|m|million)?", "polymethacrylate", lambda m: float(m.group(1)) * ({"k":1000,"m":1_000_000,"million":1_000_000}.get((m.group(2) or "").lower(), 1))),
    (r"peg(\d+(?:[.]\d+)?)(k|m|million)?", "peg", lambda m: float(m.group(1)) * ({"k":1000,"m":1_000_000,"million":1_000_000}.get((m.group(2) or "").lower(), 1))),
    (r"d2o", "d2o", None), (r"rib", "ribose", None), (r"xyl", "xylose", None),
    (r"sac|sacc|sucrose", "sucrose", None), (r"raff", "raffinose", None), (r"tol", "toluene", None),
]



def detect_standard(filename: str) -> tuple[str, Optional[float]]:
    """
    Try to detect the standard type and size from a filename.

    Returns
    -------
    (key, mw_or_none)  where key is one of STANDARD_KEYS.
    Returns ("ps", None) as fallback.
    """
    stem = os.path.splitext(os.path.basename(filename))[0].lower()
    # Search the original stem. Never remove separators: doing so turns
    # acquisition suffixes such as ``C6_1`` into ``C61``.
    for pattern, key, extractor in _DETECT_PATTERNS:
        m = re.search(pattern, stem, re.IGNORECASE)
        if m:
            mw = None
            if extractor is not None:
                try:
                    mw = extractor(m)
                except Exception:
                    mw = None
            return key, mw

    return "", None  # not detected


import os  # noqa: E402  (needed by detect_standard above)
