import json
import shutil
import subprocess

import pytest

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def make_clip(path, size="160x90", seconds=1, rate=10, audio=True):
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "lavfi", "-i", f"testsrc=size={size}:rate={rate}:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-shortest"]
    # yuv444p so odd test sizes are allowed.
    cmd += ["-pix_fmt", "yuv444p", str(path)]
    subprocess.run(cmd, check=True)
    return path


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    streams = json.loads(out)["streams"]
    video = next(s for s in streams if s["codec_type"] == "video")
    return {
        "frames": int(video["nb_read_frames"]),
        "size": (video["width"], video["height"]),
        "audio": any(s["codec_type"] == "audio" for s in streams),
    }
