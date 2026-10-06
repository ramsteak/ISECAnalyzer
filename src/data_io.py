"""
data_io.py
==========
File import, format auto-detection, and rich per-column metadata.

Column model
------------
Each column in a file has:
  - raw_name  : the original header string (or "Col 0", "Col 1", …)
  - col_type  : one of COL_TYPES  ("signal" | "time" | "weight" | "volume" | "other")
  - unit      : free string from UNIT_OPTIONS[col_type], e.g. "min", "ml", "g"

The FileData keeps a list of ColumnInfo objects (one per column).
Analysis code always refers to columns by index; the ColumnInfo provides
type/unit context for display and conversion.
"""

from __future__ import annotations
import os
import re
import io
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column type / unit constants
# ---------------------------------------------------------------------------

COL_TYPES = ["signal", "time", "weight", "volume", "other"]

UNIT_OPTIONS: dict[str, list[str]] = {
    "signal":  ["—", "mV", "mAU", "V", "A", "arb."],
    "time":    ["min", "s", "h"],
    "weight":  ["g", "mg", "kg"],
    "volume":  ["ml", "µl", "l"],
    "other":   ["—"],
}

# Default unit for each column type
DEFAULT_UNIT: dict[str, str] = {
    "signal": "mV",
    "time":   "s",
    "weight": "g",
    "volume": "ml",
    "other":  "—",
}

# Conversion factors → ml  (target unit for analysis)
_TO_ML: dict[str, float] = {
    "ml": 1.0, "µl": 1e-3, "l": 1e3,
}
# Conversion factors → minutes
_TO_MIN: dict[str, float] = {
    "min": 1.0, "s": 1/60.0, "h": 60.0,
}
# Conversion factors → grams
_TO_G: dict[str, float] = {
    "g": 1.0, "mg": 1e-3, "kg": 1e3,
}

# Common ISEC solvents:  (name, density g/ml)
SOLVENTS: list[tuple[str, float | None]] = [
    ("Custom",  None),
    ("THF",     0.889),
    ("Water",   0.998),
    ("DMF",     0.944),
    ("Toluene", 0.867),
    ("CHCl₃",   1.492),
    ("DCM",     1.325),
    ("DMAc",    0.937),
    ("0.12M Na2SO4", 1.01),
]
SOLVENT_NAMES = [s[0] for s in SOLVENTS]


# ---------------------------------------------------------------------------
# ColumnInfo
# ---------------------------------------------------------------------------

@dataclass
class ColumnInfo:
    raw_name: str           # original header or "Col N"
    col_type: str = "other" # one of COL_TYPES
    unit: str     = "—"     # from UNIT_OPTIONS[col_type]

    def display_name(self) -> str:
        return self.raw_name

    def unit_to_ml_factor(self) -> Optional[float]:
        """Return conversion factor to ml, or None if not applicable."""
        return _TO_ML.get(self.unit)

    def unit_to_min_factor(self) -> Optional[float]:
        return _TO_MIN.get(self.unit)

    def unit_to_g_factor(self) -> Optional[float]:
        return _TO_G.get(self.unit)


# ---------------------------------------------------------------------------
# Format sniffing
# ---------------------------------------------------------------------------

def _sniff_separator(sample: str) -> str:
    counts = {"\t": sample.count("\t"), ";": sample.count(";"),
              ",": sample.count(","), " ": sample.count(" ")}
    best = max(counts, key=counts.get)
    if best == " " and any(counts[s] > 0 for s in ("\t", ";", ",")):
        counts.pop(" ")
        best = max(counts, key=counts.get)
    return best


def _sniff_decimal(sample: str) -> str:
    dot = len(re.findall(r"\d+\.\d+", sample))
    com = len(re.findall(r"\d+,\d+", sample))
    return "," if com > dot else "."


def _has_header(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    first = df.iloc[0]
    for val in first:
        v = str(val).strip()
        try:
            float(v.replace(",", "."))
        except ValueError:
            return True
    return False


# ---------------------------------------------------------------------------
# Column type auto-detection from header names
# ---------------------------------------------------------------------------

_TYPE_KEYWORDS: dict[str, list[str]] = {
    "time":   ["time", "t", "zeit", "temps", "min", "sec", "hour"],
    "weight": ["weight", "mass", "masse", "gewicht", "poids", "w", "m"],
    "volume": ["volume", "vol", "v", "elution"],
    "signal": ["signal", "response", "detector", "mv", "mau", "intensity",
               "absorbance", "refractive", "ri", "uv", "dri"],
}

def _guess_col_type(name: str) -> str:
    low = name.lower().strip()
    for ctype, kws in _TYPE_KEYWORDS.items():
        for kw in kws:
            if kw in low:
                return ctype
    return "other"


def _guess_from_data(col: np.ndarray, n_cols: int, col_idx: int) -> str:
    """Fallback type guess from data shape."""
    if len(col) < 2:
        return "other"
    diffs = np.diff(col)
    mono = np.sum(diffs > 0) / len(diffs)
    if mono > 0.85:        # strongly monotone increasing → likely time/volume/weight
        return "time" if col_idx == 0 else "volume"
    if np.ptp(col) > 0 and mono < 0.6:
        return "signal"
    return "other"


# ---------------------------------------------------------------------------
# FileData
# ---------------------------------------------------------------------------

class FileData:
    """All raw data + settings for one chromatographic run file."""

    def __init__(self, path: str):
        self.path     = path
        self.filename = os.path.basename(path)

        # Format
        self.separator:  str  = " "
        self.decimal:    str  = "."
        self.has_header: bool = False

        # Raw dataframe
        self.df_raw: Optional[pd.DataFrame] = None

        # Rich column metadata  (one per column in df_raw)
        self.col_info: list[ColumnInfo] = []

        # Active columns for analysis
        self.plot_x_col: int = 0    # column shown on x-axis in main plot
        self.plot_y_col: int = 1    # column shown on y-axis in main plot
        self.analysis_x_col: int = 0  # column used as x in analysis (should be volume)
        self.analysis_y_col: int = 1

        # Signal pre-processing
        self.x_reset:   bool  = False
        self.invert_y:  bool  = False
        self.x_min:     Optional[float] = None
        self.x_max:     Optional[float] = None

        # Baseline correction
        self.baseline_enabled: bool  = False
        self.baseline_x_col:   int   = 0     # which x column the range is in
        self.baseline_x_min:   Optional[float] = None
        self.baseline_x_max:   Optional[float] = None
        self.baseline_degree:  int   = 1

        # Smoothing
        self.smoothing_enabled: bool = False
        self.smoothing_window:  int  = 11

        # Peak fitting
        self.fit_enabled: bool = False
        self.fit_model:   str  = "gaussian"

        # Momentum / elution volume
        self.momentum_degree:    int   = 1
        self.momentum_threshold: float = 75.0   # % of peak height

        # Unit conversion
        self.density:       float = 1.0    # deprecated compatibility field; experiment settings own this
        self.flow_rate:     float = 1.0    # deprecated compatibility field; experiment settings own this
        self.weight_correct: bool = False   # use regression to get flow rate

        # Standard
        self.std_key:   str            = ""    # "" = not detected
        self.std_value: Optional[float] = None
        self.rh:        Optional[float] = None

        # Results
        self.elution_volume: Optional[float] = None

        self._load()

    # ------------------------------------------------------------------
    def _load(self):
        with open(self.path, "r", encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
        sample = raw[:2000]
        self.separator = _sniff_separator(sample)
        self.decimal   = _sniff_decimal(sample)
        self._reload()

    def _reload(self, sep: str = None, dec: str = None, has_header: bool = None):
        if sep is not None:
            self.separator = sep
        if dec is not None:
            self.decimal = dec

        raw_path = self.path
        with open(raw_path, "r", encoding="utf-8", errors="replace") as fh:
            raw = fh.read()

        sep = self.separator
        dec = self.decimal

        if dec == ",":
            raw = re.sub(r"(?<=\d),(?=\d)", ".", raw)

        sep_re = sep if sep != " " else r"\s+"

        try:
            df_probe = pd.read_csv(io.StringIO(raw), sep=sep_re,
                                   engine="python", header=None)
        except Exception:
            df_probe = pd.DataFrame()

        if has_header is None:
            has_header = _has_header(df_probe)
        self.has_header = has_header

        try:
            df = pd.read_csv(io.StringIO(raw), sep=sep_re, engine="python",
                             header=0 if has_header else None)
        except Exception as exc:
            raise IOError(f"Cannot parse {self.filename}: {exc}")

        df = df.apply(pd.to_numeric, errors="coerce")
        df.dropna(how="all", inplace=True)
        df.dropna(axis=1, how="all", inplace=True)
        df.reset_index(drop=True, inplace=True)

        self.df_raw = df

        # Build / update col_info list
        n = df.shape[1]
        raw_names = [str(c) for c in df.columns]
        new_info: list[ColumnInfo] = []
        for i, rn in enumerate(raw_names):
            # Preserve existing user assignments if column count unchanged
            if i < len(self.col_info) and len(self.col_info) == n:
                ci = self.col_info[i]
                ci.raw_name = rn
                new_info.append(ci)
            else:
                # Auto-detect
                if has_header and not rn.startswith("Unnamed") and not re.match(r"^\d+$", rn):
                    ctype = _guess_col_type(rn)
                else:
                    arr = df.iloc[:, i].dropna().values.astype(float)
                    ctype = _guess_from_data(arr, n, i)
                unit = DEFAULT_UNIT.get(ctype, "—")
                new_info.append(ColumnInfo(raw_name=rn, col_type=ctype, unit=unit))
        self.col_info = new_info

        self._auto_select_cols()
        self._normalize_axes()

    def _auto_select_cols(self):
        df = self.df_raw
        if df is None or df.shape[1] < 1:
            return
        ncol = df.shape[1]

        # Find first volume/time col → analysis_x
        for i, ci in enumerate(self.col_info):
            if ci.col_type in ("volume", "time"):
                self.analysis_x_col = i
                self.plot_x_col     = i
                break
        else:
            # most monotone column
            best, bscore = 0, -1
            for i in range(ncol):
                col = df.iloc[:, i].dropna().values
                if len(col) < 2:
                    continue
                diffs = np.diff(col)
                score = abs(np.sum(diffs > 0) / len(diffs) - 0.5) * 2
                if score > bscore:
                    bscore, best = score, i
            self.analysis_x_col = best
            self.plot_x_col     = best

        # Find first signal col → y
        for i, ci in enumerate(self.col_info):
            if ci.col_type == "signal":
                self.analysis_y_col = i
                self.plot_y_col     = i
                break
        else:
            # largest range column that is not x
            best_y, best_r = 0, -1
            for i in range(ncol):
                if i == self.analysis_x_col:
                    continue
                col = df.iloc[:, i].dropna().values.astype(float)
                r = np.ptp(col) if len(col) > 0 else 0
                if r > best_r:
                    best_r, best_y = r, i
            self.analysis_y_col = best_y
            self.plot_y_col     = best_y

        self.baseline_x_col = self.analysis_x_col

    def _normalize_axes(self):
        """Repair invalid legacy axis selections, especially signal-vs-signal/time."""
        from .state import normalize_analysis_axes
        normalize_analysis_axes(self)

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get_col(self, idx: int) -> np.ndarray:
        """Return raw column values (no transforms)."""
        if self.df_raw is None or idx >= self.df_raw.shape[1]:
            return np.array([])
        return self.df_raw.iloc[:, idx].values.astype(float)

    def get_col_in_ml(self, idx: int) -> np.ndarray:
        """
        Return column values converted to ml (for volume columns) or
        to volume via flow-rate (for time columns) or via density (weight).
        Falls back to raw values for signal/other.
        """
        raw = self.get_col(idx)
        ci  = self.col_info[idx] if idx < len(self.col_info) else None
        if ci is None:
            return raw
        if ci.col_type == "volume":
            f = _TO_ML.get(ci.unit, 1.0)
            return raw * f
        if ci.col_type == "time":
            f = _TO_MIN.get(ci.unit, 1.0)
            t_min = raw * f
            return t_min * self.flow_rate
        if ci.col_type == "weight":
            f = _TO_G.get(ci.unit, 1.0)
            return (raw * f) / self.density
        return raw

    def get_xy(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (x_analysis, y) arrays with pre-processing applied.
        x is the raw values from analysis_x_col — no implicit unit conversion.
        Unit conversion (time/weight → volume) is an explicit user action that
        patches df_raw via _apply_conversion; get_xy always reads from df_raw.
        """
        if self.df_raw is None:
            return np.array([]), np.array([])

        x = self.get_col(self.analysis_x_col).astype(float)
        y = self.get_col(self.analysis_y_col).astype(float)

        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        if self.x_reset and len(x) > 0:
            x = x - x[0]
        if self.invert_y:
            y = -y
        if self.x_min is not None:
            m = x >= self.x_min; x, y = x[m], y[m]
        if self.x_max is not None:
            m = x <= self.x_max; x, y = x[m], y[m]
        return x, y

    def get_plot_xy(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the user-chosen plot columns (raw, no conversion)."""
        if self.df_raw is None:
            return np.array([]), np.array([])
        x = self.get_col(self.plot_x_col).astype(float)
        y = self.get_col(self.plot_y_col).astype(float)
        mask = np.isfinite(x) & np.isfinite(y)
        return x[mask], y[mask]

    def col_label(self, idx: int) -> str:
        """Human-readable label for a column: 'Name (unit)'."""
        if idx >= len(self.col_info):
            return f"Col {idx}"
        ci = self.col_info[idx]
        unit = ci.unit if ci.unit != "—" else ""
        return f"{ci.raw_name} ({unit})" if unit else ci.raw_name

    def volume_col_index(self) -> Optional[int]:
        """Return the index of the first column whose type is 'volume', or None."""
        for i, ci in enumerate(self.col_info):
            if ci.col_type == "volume":
                return i
        return None

    def add_derived_column(self, data: "np.ndarray", name: str,
                           col_type: str, unit: str) -> int:
        """
        Append a new derived column to df_raw and col_info.
        Returns the index of the new column.
        Existing columns are untouched.
        """
        import pandas as pd
        if self.df_raw is None:
            raise RuntimeError("No data loaded")
        # Use a unique column name so repeated calls don't clash
        col_name = name
        existing = list(self.df_raw.columns)
        if col_name in existing:
            n = 1
            while f"{col_name}_{n}" in existing:
                n += 1
            col_name = f"{col_name}_{n}"
        self.df_raw[col_name] = data
        self.col_info.append(ColumnInfo(raw_name=name, col_type=col_type, unit=unit))
        return len(self.col_info) - 1

    def get_display_name(self) -> str:
        return self.filename


# ---------------------------------------------------------------------------
# Unit conversion helpers (used by GUI conversion action)
# ---------------------------------------------------------------------------

def weight_to_volume(weight_g: np.ndarray, density: float) -> np.ndarray:
    return weight_g / density


def time_to_volume(time_min: np.ndarray, flow_rate: float) -> np.ndarray:
    return time_min * flow_rate


def weight_correct_time(time_min: np.ndarray, weight_g: np.ndarray,
                        density: float) -> tuple[np.ndarray, float]:
    from scipy import stats
    slope, intercept, *_ = stats.linregress(time_min, weight_g)
    flow_rate = slope / density
    volume = time_min * slope / density
    return volume, float(flow_rate)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_elution_volumes(file_data_list: list[FileData], output_path: str):
    rows = []
    for fd in file_data_list:
        rows.append({
            "Filename": fd.filename,
            "Standard": fd.std_key,
            "MW_or_nC": fd.std_value,
            "Rh_nm":    fd.rh,
            "Ve_ml":    fd.elution_volume,
        })
    df = pd.DataFrame(rows)
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".xlsx":
        df.to_excel(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

# ---------------------------------------------------------------------------
# Robust filename metadata parsing
# ---------------------------------------------------------------------------
def parse_sample_size(filename: str):
    """Extract standard metadata without consuming acquisition suffixes."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    for key, aliases in (("ps", ["ps", "polystyrene"]),
                         ("peg", ["peg", "peo"]),
                         ("dextran", ["dextran", "dex"]),
                         ("polymethacrylate", ["pmma", "polymethacrylate"])):
        alias = "(?:" + "|".join(map(re.escape, aliases)) + ")"
        m = re.search(rf"(?<![A-Za-z0-9]){alias}\s*([0-9]+(?:[.,][0-9]+)?)\s*(kda|da|k|m)?(?=$|[^0-9A-Za-z])", stem, re.I)
        if m:
            value = float(m.group(1).replace(',', '.'))
            unit = (m.group(2) or '').lower()
            if unit in ('k', 'kda'): value *= 1000
            elif unit == 'm': value *= 1_000_000
            return key, value
    # The alkane token ends at the first non-digit. Thus C6_1 is C6, never C61.
    m = re.search(r"(?<![A-Za-z0-9])c\s*([0-9]{1,3})(?![0-9])", stem, re.I)
    if m:
        return "alkane", float(m.group(1))
    return None, None

