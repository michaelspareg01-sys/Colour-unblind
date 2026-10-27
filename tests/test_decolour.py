import cv2
import numpy as np
import pytest

from colour_unblind.colourspace import bgr_to_linear, linear_to_bgr, luminance
from colour_unblind.decolour import ContrastPreservingGrey, _evaluate, fit_decolor


def scene(width=256, height=144, seed=1):
    rng = np.random.default_rng(seed)
    small = (rng.random((9, 16, 3)) * 255).astype(np.uint8)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)


def test_fit_reproduces_opencv_decolor():
    frame = scene()
    expected, _ = cv2.decolor(frame)
    coef = fit_decolor(frame)
    fitted = _evaluate(np.clip(frame, 1, 254)[..., ::-1].astype(np.float32) / 255, coef)
    assert np.corrcoef(fitted.ravel(), expected.ravel())[0, 1] > 0.99


def test_output_is_grey():
    out = ContrastPreservingGrey().apply(bgr_to_linear(scene()))
    assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 0], out[..., 2])


def test_full_resolution_frame_matches_analysis_shape():
    frame = scene(1280, 720)
    out = ContrastPreservingGrey().apply(bgr_to_linear(frame))
    assert out.shape == (720, 1280, 3)


def test_flat_two_colour_frame_falls_back_to_luminance():
    # cv2.decolor degenerates (all zeros) on a few flat colours; use CIE Y then.
    img = np.zeros((64, 128, 3), dtype=np.float32)
    img[:, :64] = (1.0, 0.0, 0.0)
    img[:, 64:] = (0.0, 0.0, 1.0)
    out = ContrastPreservingGrey().apply(img)
    np.testing.assert_allclose(out[0, 0], [0.2126] * 3, atol=1e-6)
    np.testing.assert_allclose(out[0, 127], [0.0722] * 3, atol=1e-6)


def test_isoluminant_patches_in_a_scene_get_different_greys():
    img = np.zeros((128, 256, 3), dtype=np.float32)
    img[:] = np.linspace(0, 1, 256, dtype=np.float32)[None, :, None]
    img[32:96, 32:96] = (1.0, 0.0, 0.0)
    img[32:96, 160:224] = (0.0, 0.2126 / 0.7152, 0.0)
    out = ContrastPreservingGrey().apply(img)
    assert abs(float(out[64, 64, 0]) - float(out[64, 192, 0])) > 0.02


def test_flat_frame_does_not_blow_up():
    img = np.full((50, 80, 3), 0.3, dtype=np.float32)
    out = ContrastPreservingGrey().apply(img)
    assert np.all(np.isfinite(out))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_smoothing_must_be_in_range():
    with pytest.raises(ValueError):
        ContrastPreservingGrey(smoothing=0.0)


def test_factored_polynomial_matches_feature_form():
    from colour_unblind.decolour import _evaluate, _features
    rng = np.random.default_rng(3)
    rgb = rng.random((20, 30, 3), dtype=np.float32)
    coef = rng.normal(size=10).astype(np.float32)
    np.testing.assert_allclose(_evaluate(rgb, coef), _features(rgb) @ coef, rtol=1e-5, atol=1e-5)


def test_refits_only_every_few_frames(monkeypatch):
    grey = ContrastPreservingGrey(refit_every=3)
    calls = []
    real_fit = grey._fit
    monkeypatch.setattr(grey, "_fit", lambda bgr: calls.append(1) or real_fit(bgr))
    frame = bgr_to_linear(scene())
    for _ in range(7):
        grey.apply(frame)
    assert len(calls) == 3  # frames 1, 4 and 7


def test_720p_frame_is_fast_enough_for_video():
    import time
    grey = ContrastPreservingGrey()
    frame = bgr_to_linear(scene(1280, 720))
    grey.apply(frame)
    start = time.perf_counter()
    for _ in range(6):
        grey.apply(frame)
    assert (time.perf_counter() - start) / 6 < 0.25


def test_inverted_decolor_result_is_flipped_back(monkeypatch):
    # cv2.decolor can return an image whose brightness runs the wrong way.
    from colour_unblind.decolour import _encoded_luma

    def inverted(bgr):
        grey = np.round(_encoded_luma(bgr) * 255).astype(np.uint8)
        return 255 - grey, bgr

    monkeypatch.setattr("colour_unblind.decolour.cv2.decolor", inverted)
    # (decolor's grey is then fully explained by luminance, so output = luminance)
    frame = scene()
    out = ContrastPreservingGrey().apply(bgr_to_linear(frame))
    expected = luminance(bgr_to_linear(frame))
    assert np.corrcoef(out[..., 0].ravel(), expected.ravel())[0, 1] > 0.98


def test_dark_scene_stays_dark():
    frame = (scene(320, 180).astype(np.float32) * 0.15).astype(np.uint8)  # a night scene
    luma = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out = linear_to_bgr(ContrastPreservingGrey().apply(bgr_to_linear(frame)))[..., 0]
    assert abs(float(out.mean()) - float(luma.mean())) < 15


def test_white_and_black_stay_near_white_and_black_in_a_mixed_scene():
    frame = scene(320, 180)
    frame[20:60, 20:60] = 255
    frame[100:140, 20:60] = 0
    out = linear_to_bgr(ContrastPreservingGrey().apply(bgr_to_linear(frame)))[..., 0]
    assert out[30:50, 30:50].mean() > 215
    assert out[110:130, 30:50].mean() < 40
