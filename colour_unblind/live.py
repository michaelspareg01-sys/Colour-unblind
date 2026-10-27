"""Live webcam view. Keys: m = next mode, v = next viewer, q or Esc = quit."""

import cv2

from colour_unblind.processor import MODES, VIEWER_NAMES, VIEWERS, FrameProcessor

WINDOW = "Colour Unblind"
_ESC = 27


def open_camera(index: int = 0) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera {index}; try --camera 1, or check camera permissions")
    return cap


def _next(options: tuple, current: str) -> str:
    return options[(options.index(current) + 1) % len(options)]


def _status(frame, processor: FrameProcessor) -> None:
    text = f"{processor.mode} | {VIEWER_NAMES[processor.viewer]} | m: mode  v: viewer  q: quit"
    h = frame.shape[0]
    scale = max(0.4, h / 900)
    (_, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    frame[h - th - base - 8:] //= 3
    cv2.putText(frame, text, (6, h - base - 4), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), 1, cv2.LINE_AA)


def _show_in_window(frame) -> None:
    cv2.imshow(WINDOW, frame)


def _wait_key_in_window() -> int:
    key = cv2.waitKey(1)
    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
        return ord("q")  # window closed with the mouse
    return key


def run_live(processor: FrameProcessor, capture, show=None, wait_key=None) -> FrameProcessor:
    """Show processed camera frames until the camera stops or the user quits.
    Returns the processor in its final mode/viewer."""
    windowed = show is None
    show = show or _show_in_window
    wait_key = wait_key or _wait_key_in_window
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            out = processor.process(frame)
            _status(out, processor)
            show(out)
            key = wait_key() & 0xFF
            if key in (ord("q"), _ESC):
                break
            if key == ord("m"):
                processor = processor.with_mode(_next(MODES, processor.mode))
            elif key == ord("v"):
                processor = processor.with_viewer(_next(VIEWERS, processor.viewer))
    finally:
        capture.release()
        if windowed:
            cv2.destroyAllWindows()
    return processor
