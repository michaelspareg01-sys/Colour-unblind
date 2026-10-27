import os
import stat
import subprocess
import threading

import cv2
import numpy as np
import pytest

from colour_unblind.processor import FrameProcessor
from colour_unblind.video import process_video
from conftest import make_clip, needs_ffmpeg, probe


def duration(path, stream):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", stream, "-show_entries", "stream=duration",
         "-of", "csv=p=0", str(path)], check=True, capture_output=True, text=True).stdout.strip()
    return float(out)


@needs_ffmpeg
def test_noisy_ffmpeg_stderr_does_not_deadlock(tmp_path, monkeypatch):
    fake = tmp_path / "ffmpeg"
    fake.write_text('#!/bin/sh\nfor a; do last="$a"; done\n'
                    "head -c 2000000 /dev/zero | tr '\\0' x >&2\n"
                    'cat > "$last"\nexit 0\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    src = make_clip(tmp_path / "in.mp4")
    monkeypatch.setattr("colour_unblind.video.shutil.which", lambda name: str(fake))
    done = []
    t = threading.Thread(target=lambda: done.append(
        process_video(src, tmp_path / "out.mp4", FrameProcessor("cat", "simulate"), progress=False)),
        daemon=True)
    t.start()
    t.join(30)
    assert done == [10]


@needs_ffmpeg
def test_ffmpeg_error_message_reaches_the_user_and_no_partial_file(tmp_path):
    src = make_clip(tmp_path / "in.mp4")
    out = tmp_path / "out.notavideoformat"
    with pytest.raises(RuntimeError, match="ffmpeg failed: .+"):
        process_video(src, out, FrameProcessor("cat", "simulate"), progress=False)
    assert list(tmp_path.iterdir()) == [src]


def test_missing_output_folder_is_a_clear_error(tmp_path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="output folder"):
        process_video(src, tmp_path / "nope" / "out.mp4", FrameProcessor("cat", "simulate"))


@needs_ffmpeg
def test_processing_error_leaves_no_output_and_keeps_its_own_message(tmp_path):
    class Boom(FrameProcessor):
        def process(self, frame):
            raise KeyError("boom")

    src = make_clip(tmp_path / "in.mp4")
    out = tmp_path / "out.mp4"
    with pytest.raises(KeyError, match="boom"):
        process_video(src, out, Boom("cat", "simulate"), progress=False)
    assert not out.exists()
    assert list(tmp_path.iterdir()) == [src]


@needs_ffmpeg
def test_output_is_tagged_bt709_and_colours_survive(tmp_path):
    src = tmp_path / "red.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "color=c=0xFB1010:size=320x180:rate=10:duration=1",
                    "-vf", "scale=out_color_matrix=bt709:out_range=tv", "-colorspace", "bt709",
                    "-color_primaries", "bt709", "-color_trc", "bt709", "-pix_fmt", "yuv420p", str(src)],
                   check=True)
    out = tmp_path / "out.mp4"
    process_video(src, out, FrameProcessor("deutan", "recolour", strength=0.0), progress=False)
    tags = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
                           "stream=color_space,color_primaries,color_transfer", "-of", "csv=p=0", str(out)],
                          check=True, capture_output=True, text=True).stdout.strip()
    assert tags == "bt709,bt709,bt709"
    a = cv2.VideoCapture(str(src)).read()[1].astype(int)
    b = cv2.VideoCapture(str(out)).read()[1].astype(int)
    assert np.abs(a[40:140, 60:260] - b[40:140, 60:260]).max() <= 6


@needs_ffmpeg
def test_variable_frame_rate_keeps_duration_and_audio(tmp_path):
    src = tmp_path / "vfr.mkv"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", "testsrc=size=160x90:rate=30:duration=2",
                    "-f", "lavfi", "-i", "testsrc=size=160x90:rate=5:duration=2",
                    "-f", "lavfi", "-i", "sine=duration=4",
                    "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]", "-map", "2:a",
                    "-fps_mode", "vfr", "-pix_fmt", "yuv420p", str(src)], check=True)
    out = tmp_path / "out.mp4"
    process_video(src, out, FrameProcessor("cat", "simulate"), progress=False)
    assert abs(duration(out, "v:0") - 4.0) < 0.3
    assert abs(duration(out, "a:0") - 4.0) < 0.2
