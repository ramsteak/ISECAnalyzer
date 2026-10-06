"""Application state model for ISEC Analyzer.

The GUI should treat this module as the single source of truth. Raw files contain
only data and file-local metadata; experiment-wide conversion settings live in
GlobalSettings, while processing options live in AnalysisSettings.
"""
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class GlobalSettings:
    solvent: str = ""
    density_g_ml: float = 1.0
    flow_rate_ml_min: float = 1.0
    time_unit: str = "s"
    weight_correct: bool = False

    def validate(self):
        if self.density_g_ml <= 0:
            raise ValueError("Density must be greater than zero.")
        if self.flow_rate_ml_min <= 0 and not self.weight_correct:
            raise ValueError("Flow rate must be greater than zero.")

@dataclass
class AnalysisSettings:
    x_col: int = 0
    y_col: int = 1
    x_reset: bool = False
    invert_y: bool = False
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    baseline_enabled: bool = False
    baseline_x_col: int = 0
    baseline_x_min: Optional[float] = None
    baseline_x_max: Optional[float] = None
    baseline_degree: int = 1
    smoothing_enabled: bool = False
    smoothing_window: int = 11
    fit_enabled: bool = False
    fit_model: str = "gaussian"
    momentum_degree: int = 1
    momentum_threshold: float = 75.0

@dataclass
class AppState:
    global_settings: GlobalSettings = field(default_factory=GlobalSettings)
    files: list = field(default_factory=list)
    selected_indices: list[int] = field(default_factory=list)


def valid_signal_columns(file_data):
    """Return indices that are legal Y-axis choices."""
    return [i for i, c in enumerate(file_data.col_info) if c.col_type == "signal"]


def normalize_analysis_axes(file_data):
    """Repair stale settings loaded from old sessions/files.

    X must be a coordinate-like column and Y must be a signal. This function is
    deliberately deterministic and is safe to call after every metadata change.
    """
    n = len(file_data.col_info)
    if n == 0:
        file_data.analysis_x_col = file_data.analysis_y_col = 0
        return
    x_candidates = [i for i, c in enumerate(file_data.col_info)
                    if c.col_type in {"time", "volume", "weight"}]
    if file_data.analysis_x_col not in x_candidates:
        file_data.analysis_x_col = x_candidates[0] if x_candidates else 0
    signals = valid_signal_columns(file_data)
    if file_data.analysis_y_col not in signals:
        non_x = [i for i in signals if i != file_data.analysis_x_col]
        file_data.analysis_y_col = (non_x or signals or
                                    [i for i in range(n) if i != file_data.analysis_x_col] or [0])[0]
    file_data.baseline_x_col = file_data.analysis_x_col
