from isec.standards import detect_standard

def test_acquisition_suffix_not_sample_size():
    assert detect_standard('C6_1.dat') == ('alkane', 6.0)
    assert detect_standard('C5_5.dat') == ('alkane', 5.0)
    assert detect_standard('C5_4_abemus.dat') == ('alkane', 5.0)
