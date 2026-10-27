"""Read a video file, process every frame, write it back out with its audio."""

import shutil
import subprocess
import sys
from pathlib import Path

import cv2

from colour_unblind.processor import FrameProcessor


class _FfmpegWriter:
    """Pipes raw BGR frames into ffmpeg and copies the source's audio track."""

    def __init__(self, ffmpeg: str, source: Path, output: Path, size: tuple[int, int], fps: float):
        w, h = size
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.6g}", "-i", "-",
            "-i", str(source),
            "-map", "0:v:0", "-map", "1:a?",
            # yuv420p plays everywhere but needs even sizes, so trim a pixel if odd.
            "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p",
            "-shortest", str(output),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame) -> None:
        try:
            self.proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            self.close()

    def close(self) -> None:
        if self.proc.stdin and not self.proc.stdin.closed:
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
        err = self.proc.stderr.read().decode(errors="replace")
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {err.strip()}")


class _OpenCvWriter:
    def __init__(self, output: Path, size: tuple[int, int], fps: float):
        self.writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self.writer.isOpened():
            raise RuntimeError(f"OpenCV could not open {output} for writing")

    def write(self, frame) -> None:
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()


def process_video(source, output, processor: FrameProcessor, progress: bool = True) -> int:
    """Process every frame of ``source`` into ``output``. Returns the frame count."""
    source, output = Path(source), Path(output)
    if not source.is_file():
        raise FileNotFoundError(f"no such video file: {source}")
    if source.resolve() == output.resolve():
        raise ValueError("output must be a different file from the input")

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"could not read video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise ValueError(f"no frames in video: {source}")
    size = (frame.shape[1], frame.shape[0])

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        writer = _FfmpegWriter(ffmpeg, source, output, size, fps)
    else:
        print("warning: ffmpeg not found, writing video without audio", file=sys.stderr)
        writer = _OpenCvWriter(output, size, fps)

    count = 0
    try:
        while ok:
            writer.write(processor.process(frame))
            count += 1
            if progress and count % 10 == 0:
                of = f"/{total}" if total > 0 else ""
                print(f"\rprocessed {count}{of} frames", end="", file=sys.stderr, flush=True)
            ok, frame = cap.read()
    finally:
        cap.release()
        writer.close()
    if progress:
        print(f"\rprocessed {count} frames -> {output}", file=sys.stderr)
    return count
