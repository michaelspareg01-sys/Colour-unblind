"""Read a video file, process every frame, write it back out with its audio."""

import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

from colour_unblind.processor import FrameProcessor


class _FfmpegWriter:
    """Pipes raw BGR frames into ffmpeg (H.264 + AAC) and copies the source's audio."""

    def __init__(self, ffmpeg: str, source: Path, output: Path, size: tuple[int, int], fps: float):
        w, h = size
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.6g}", "-i", "-",
            "-i", str(source),
            "-map", "0:v:0", "-map", "1:a?",
            # yuv420p plays everywhere but needs even sizes, so trim a pixel if
            # odd. Convert and tag as BT.709, which players assume for HD.
            "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2,scale=out_color_matrix=bt709:out_range=tv",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-c:a", "aac",
            str(output),
        ]
        # A file, not a pipe: ffmpeg can log a lot (e.g. a damaged audio
        # track) and a full pipe nobody reads would deadlock both processes.
        self._log = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self._log)
        self._closed = False

    def write(self, frame) -> None:
        try:
            self.proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            self.close()  # raises with ffmpeg's own message if it failed
            raise RuntimeError("ffmpeg stopped reading frames early") from None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        code = self.proc.wait()
        self._log.seek(0)
        err = self._log.read().decode(errors="replace").strip()
        self._log.close()
        if code != 0:
            raise RuntimeError(f"ffmpeg failed: {err or f'exit code {code}'}")

    def abort(self) -> None:
        if not self._closed:
            self.proc.kill()
            try:
                self.close()
            except RuntimeError:
                pass


class _OpenCvWriter:
    def __init__(self, output: Path, size: tuple[int, int], fps: float):
        self.writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self.writer.isOpened():
            raise RuntimeError(f"OpenCV could not open {output} for writing")

    def write(self, frame) -> None:
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()

    abort = close


def _frame_rate(cap) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps if math.isfinite(fps) and 0 < fps <= 240 else 30.0


def process_video(source, output, processor: FrameProcessor, progress: bool = True) -> int:
    """Process every frame of ``source`` into ``output``. Returns frames written.

    Output is constant frame rate: frames are placed by their timestamps, so
    variable-frame-rate sources (phones, screen recorders) stay in sync with
    their audio. Output appears only when everything succeeded."""
    source, output = Path(source), Path(output)
    if not source.is_file():
        raise FileNotFoundError(f"no such video file: {source}")
    if source.resolve() == output.resolve():
        raise ValueError("output must be a different file from the input")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output folder does not exist: {output.parent}")

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"could not read video: {source}")
    try:
        return _process(cap, source, output, processor, progress)
    finally:
        cap.release()


def _process(cap, source: Path, output: Path, processor: FrameProcessor, progress: bool) -> int:
    fps = _frame_rate(cap)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = cap.read()
    if not ok:
        raise ValueError(f"no frames in video: {source}")
    if frame.shape[0] < 2 or frame.shape[1] < 2:
        raise ValueError(f"frames are too small to process: {frame.shape[1]}x{frame.shape[0]}")
    size = (frame.shape[1], frame.shape[0])

    partial = output.with_name(f".{output.stem}.partial{output.suffix}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        writer = _FfmpegWriter(ffmpeg, source, partial, size, fps)
    else:
        print("warning: ffmpeg not found, writing video without audio", file=sys.stderr)
        writer = _OpenCvWriter(partial, size, fps)

    decoded = written = 0
    start = previous = None
    try:
        while ok:
            decoded += 1
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if start is None:
                start = t
            # Fall back to "one frame per slot" if the timestamps are unusable.
            target = round((t - start) * fps) if previous is None or t > previous else written
            previous = t
            if target >= written:
                out = processor.process(frame)
                for _ in range(target - written + 1):
                    writer.write(out)
                written = target + 1
            if progress and decoded % 10 == 0:
                of = f"/{total}" if total > 0 else ""
                print(f"\rprocessed {decoded}{of} frames", end="", file=sys.stderr, flush=True)
            ok, frame = cap.read()
        writer.close()
        os.replace(partial, output)
    except BaseException:
        writer.abort()
        partial.unlink(missing_ok=True)
        raise
    if progress:
        print(f"\rprocessed {decoded} frames -> {output}", file=sys.stderr)
    return written
