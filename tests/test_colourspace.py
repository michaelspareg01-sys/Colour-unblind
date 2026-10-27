import numpy as np
import pytest

from colour_unblind.colourspace import apply_matrix, bgr_to_linear, linear_to_bgr, luminance


def test_all_grey_levels_round_trip_exactly():
    levels = np.arange(256, dtype=np.uint8)
    frame = np.stack([levels] * 3, axis=-1).reshape(16, 16, 3)
    assert np.array_equal(linear_to_bgr(bgr_to_linear(frame)), frame)


def test_srgb_128_decodes_to_iec_61966_value():
    frame = np.full((1, 1, 3), 128, dtype=np.uint8)
    assert bgr_to_linear(frame)[0, 0, 0] == pytest.approx(0.2158605, abs=1e-5)


def test_bgr_input_becomes_rgb_order():
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    frame[0, 0] = (255, 0, 0)  # pure blue in BGR
    assert bgr_to_linear(frame)[0, 0].tolist() == [0.0, 0.0, 1.0]
    assert linear_to_bgr(np.array([[[0.0, 0.0, 1.0]]]))[0, 0].tolist() == [255, 0, 0]


def test_linear_to_bgr_clips_out_of_gamut():
    out = linear_to_bgr(np.array([[[-0.5, 2.0, np.nan]]]))
    assert out[0, 0].tolist() == [0, 255, 0]


def test_luminance_uses_bt709_weights():
    img = np.eye(3, dtype=np.float32).reshape(1, 3, 3)
    assert luminance(img)[0].tolist() == pytest.approx([0.2126, 0.7152, 0.0722])


def test_apply_matrix_multiplies_each_pixel():
    img = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
    m = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float64)
    out = apply_matrix(img, m)
    assert out.dtype == np.float32
    assert out[0, 0].tolist() == [3.0, 2.0, 1.0]
