import numpy as np
from isec.pipeline import run_pipeline, PipelineConfig
from isec.data_io import parse_sample_size

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
