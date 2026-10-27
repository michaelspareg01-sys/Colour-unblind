import numpy as np
import pytest

from colour_unblind.live import open_camera, run_live
from colour_unblind.processor import FrameProcessor


class FakeCapture:
    def __init__(self, n):
        self.frames = [np.full((48, 64, 3), 25 * i, dtype=np.uint8) for i in range(n)]
        self.released = False

    def read(self):
        return (True, self.frames.pop(0)) if self.frames else (False, None)

    def release(self):
        self.released = True


def test_keys_cycle_mode_then_viewer_then_quit():
    shown, keys = [], iter([ord("m"), ord("v"), ord("q"), ord("x")])
    cap = FakeCapture(10)
    final = run_live(FrameProcessor("deutan", "simulate"), cap,
                     show=lambda frame: shown.append(frame), wait_key=lambda: next(keys))
    assert len(shown) == 3
    assert all(f.shape == (48, 64, 3) for f in shown)
    assert (final.mode, final.viewer) == ("recolour", "tritan")
    assert cap.released


def test_stops_when_camera_runs_out():
    shown = []
    run_live(FrameProcessor("cat", "compare"), FakeCapture(2),
             show=shown.append, wait_key=lambda: -1, retry_delay=0)
    assert len(shown) == 2


def test_escape_quits():
    shown = []
    run_live(FrameProcessor("cat", "simulate"), FakeCapture(5), show=shown.append, wait_key=lambda: 27)
    assert len(shown) == 1


def test_camera_that_does_not_open_is_a_clear_error():
    with pytest.raises(RuntimeError, match="camera 99"):
        open_camera(99)


class WarmingUpCapture(FakeCapture):
    """Fails its first few reads, like many webcams while they start up."""

    def __init__(self, n, failures):
        super().__init__(n)
        self.failures = failures

    def read(self):
        if self.failures:
            self.failures -= 1
            return False, None
        return super().read()


def test_waits_for_a_webcam_that_is_warming_up():
    shown = []
    run_live(FrameProcessor("cat", "simulate"), WarmingUpCapture(3, failures=5),
             show=shown.append, wait_key=lambda: -1, retry_delay=0)
    assert len(shown) == 3


def test_camera_that_never_sends_a_frame_is_a_clear_error():
    with pytest.raises(RuntimeError, match="no frames"):
        run_live(FrameProcessor("cat", "simulate"), FakeCapture(0),
                 show=lambda f: None, wait_key=lambda: -1, retry_delay=0)


def test_says_why_it_stopped_when_the_camera_stops(capsys):
    run_live(FrameProcessor("cat", "simulate"), FakeCapture(2),
             show=lambda f: None, wait_key=lambda: -1, retry_delay=0)
    assert "stopped sending frames" in capsys.readouterr().err
