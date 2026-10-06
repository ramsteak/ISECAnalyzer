import numpy as np
from isec.state import normalize_analysis_axes, GlobalSettings

class C:
    def __init__(self, t): self.col_type=t
class F:
    col_info=[C('time'), C('signal'), C('signal')]
    analysis_x_col=0; analysis_y_col=0; baseline_x_col=0

def test_invalid_y_is_repaired():
    f=F(); normalize_analysis_axes(f); assert f.analysis_y_col == 1

def test_global_settings_validation():
    s=GlobalSettings(); s.validate()
    try:
        s.density_g_ml=0; s.validate()
    except ValueError: pass
    else: raise AssertionError
