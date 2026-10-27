import cv2
import numpy as np
import pytest

from colour_unblind.processor import MODES, VIEWERS, FrameProcessor


def noisy_frame(w=320, h=180, seed=0):
    rng = np.random.default_rng(seed)
    small = (rng.random((h // 10, w // 10, 3)) * 255).astype(np.uint8)
    frame = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(frame.astype(np.int16) + rng.integers(-3, 4, frame.shape), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("viewer", VIEWERS)
@pytest.mark.parametrize("size", [(320, 180), (641, 479)])
def test_output_has_input_shape_and_type(mode, viewer, size):
    frame = noisy_frame(*size)
    out = FrameProcessor(viewer, mode).process(frame)
    assert out.shape == frame.shape and out.dtype == np.uint8


def test_compare_top_left_tile_is_the_original():
    frame = noisy_frame(641, 479)
    out = FrameProcessor("deutan", "compare").process(frame)
    tile = cv2.resize(frame, (320, 239), interpolation=cv2.INTER_AREA)
    # Skip the label strip at the top of the tile.
    assert np.array_equal(out[120:239, :320], tile[120:])


def test_compare_bottom_left_tile_is_the_recolouring():
    frame = noisy_frame(640, 480)
    out = FrameProcessor("deutan", "compare").process(frame)
    small = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
    expected = FrameProcessor("deutan", "recolour").process(small)
    assert np.abs(out[360:480, :320].astype(int) - expected[120:].astype(int)).max() <= 1


def test_blur_only_affects_simulation():
    frame = noisy_frame()
    sharp, blurry = FrameProcessor("cat", "simulate"), FrameProcessor("cat", "simulate", blur=3.0)
    assert not np.array_equal(sharp.process(frame), blurry.process(frame))
    assert np.array_equal(
        FrameProcessor("cat", "recolour").process(frame),
        FrameProcessor("cat", "recolour", blur=3.0).process(frame),
    )


def test_white_stays_white_in_every_mode():
    frame = np.full((40, 60, 3), 255, dtype=np.uint8)
    for viewer in VIEWERS:
        for mode in ("simulate", "recolour"):
            assert FrameProcessor(viewer, mode).process(frame).min() >= 254, (viewer, mode)


def test_with_viewer_and_with_mode_keep_settings():
    p = FrameProcessor("protan", "simulate", severity=0.4, strength=0.5, blur=1.0)
    q = p.with_viewer("tritan").with_mode("compare")
    assert (q.viewer, q.mode, q.severity, q.strength, q.blur) == ("tritan", "compare", 0.4, 0.5, 1.0)


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        FrameProcessor("dog", "simulate")
    with pytest.raises(ValueError):
        FrameProcessor("cat", "paint")
    with pytest.raises(ValueError):
        FrameProcessor("cat", "simulate", blur=-1)


def test_labels_use_the_right_article():
    from colour_unblind.processor import tile_titles
    assert tile_titles("achromat")[1] == "How an achromat (no colour) sees it"
    assert tile_titles("cat")[3] == "Recoloured, as a cat sees it"


def test_label_never_runs_off_a_narrow_tile():
    from colour_unblind.processor import _label
    tile = np.full((240, 200, 3), 255, dtype=np.uint8)
    _label(tile, "Recoloured, as an achromat (no colour) sees it")
    assert tile[:, -2:].min() == 255  # right edge untouched
    assert tile[:, :100].min() < 255  # but something was drawn
