import numpy as np
import pandas as pd
from isec.pipeline import run_pipeline, PipelineConfig
from isec.data_io import export_analyte_elution_data, parse_sample_size

def test_pipeline_returns_sorted_result():
    r = run_pipeline([2, 1, 3], [0, 1, 0])
    assert np.all(np.diff(r.x) > 0)
    assert r.y_corrected.shape == r.x.shape

def test_filename_parser_tolerates_noise():
    assert parse_sample_size('run 02.PS 100 kDa.final.txt') == ('ps', 100000.0)
    assert parse_sample_size('sample_C18_2026.03.csv') == ('alkane', 18.0)


def test_alkane_suffixes_are_not_part_of_name():
    assert parse_sample_size("C6_1.dat") == ("alkane", 6.0)
    assert parse_sample_size("C5_5.dat") == ("alkane", 5.0)
    assert parse_sample_size("C5_4_abemus.dat") == ("alkane", 5.0)


def test_analyte_elution_export_has_explicit_columns(tmp_path):
    fd = type("FileDataStub", (), {
        "filename": "PS100.dat",
        "std_key": "ps",
        "std_value": 100000.0,
        "rh": 5.2,
        "elution_volume": 12.34,
    })()
    output_path = tmp_path / "analytes.csv"

    export_analyte_elution_data([fd], str(output_path))

    exported = pd.read_csv(output_path)
    assert list(exported.columns) == [
        "Filename", "Analyte type", "Size",
        "Hydrodynamic radius (nm)", "Elution volume (ml)",
    ]
    assert exported.iloc[0].to_dict() == {
        "Filename": "PS100.dat",
        "Analyte type": "ps",
        "Size": 100000.0,
        "Hydrodynamic radius (nm)": 5.2,
        "Elution volume (ml)": 12.34,
    }
