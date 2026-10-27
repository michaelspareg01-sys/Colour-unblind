"""Turn one BGR frame into the requested view: simulate, recolour or compare."""

import cv2
import numpy as np

from colour_unblind.colourspace import bgr_to_linear, linear_to_bgr
from colour_unblind.recolour import recolour_for
from colour_unblind.simulate import VIEWERS, simulation_for

MODES = ("simulate", "recolour", "compare")

VIEWER_NAMES = {
    "protan": "protan (red-blind)",
    "deutan": "deutan (green-blind)",
    "tritan": "tritan (blue-blind)",
    "cat": "cat",
    "achromat": "achromat (no colour)",
}


def tile_titles(viewer: str) -> tuple[str, str, str, str]:
    """Titles of the four compare tiles, in reading order."""
    name = VIEWER_NAMES[viewer]
    a = "an" if name[0] in "aeiou" else "a"
    return ("Original", f"How {a} {name} sees it", "Recoloured", f"Recoloured, as {a} {name} sees it")


def _label(tile: np.ndarray, text: str) -> None:
    """Draw ``text`` on a dark strip across the top of ``tile``, in place,
    shrinking it to fit the tile's width."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.35, tile.shape[0] / 500)
    width = cv2.getTextSize(text, font, scale, 1)[0][0]
    scale *= min(1.0, (tile.shape[1] * 0.9) / (width * 1.1))
    thickness = max(1, round(scale * 1.5))
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    pad = max(3, th // 3)
    strip = tile[: th + base + 2 * pad, : tw + 2 * pad]
    strip[:] = strip // 3
    cv2.putText(tile, text, (pad, pad + th), font, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)


class FrameProcessor:
    """``severity`` (0-1) is for protan/deutan/tritan simulation, ``strength``
    scales the recolouring, ``blur`` is a Gaussian sigma in pixels applied to
    simulations (for cats' lower acuity)."""

    def __init__(self, viewer: str, mode: str, severity: float = 1.0,
                 strength: float = 1.0, blur: float = 0.0):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose from {', '.join(MODES)}")
        if blur < 0:
            raise ValueError(f"blur must be >= 0, got {blur}")
        self.viewer, self.mode = viewer, mode
        self.severity, self.strength, self.blur = severity, strength, blur
        self._simulation = simulation_for(viewer, severity)
        self._recolour = recolour_for(viewer, strength)

    def with_viewer(self, viewer: str) -> "FrameProcessor":
        return FrameProcessor(viewer, self.mode, self.severity, self.strength, self.blur)

    def with_mode(self, mode: str) -> "FrameProcessor":
        return FrameProcessor(self.viewer, mode, self.severity, self.strength, self.blur)

    def _simulate(self, linear: np.ndarray) -> np.ndarray:
        out = self._simulation.apply(linear)
        if self.blur > 0:
            out = cv2.GaussianBlur(out, (0, 0), self.blur)
        return out

    def process(self, frame: np.ndarray) -> np.ndarray:
        if self.mode == "compare":
            return self._compare(frame)
        linear = bgr_to_linear(frame)
        if self.mode == "simulate":
            return linear_to_bgr(self._simulate(linear))
        return linear_to_bgr(self._recolour.apply(linear))

    def _compare(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        w1, h1 = w // 2, h // 2
        small = cv2.resize(frame, (w1, h1), interpolation=cv2.INTER_AREA)
        linear = bgr_to_linear(small)
        recoloured = np.clip(self._recolour.apply(linear), 0.0, 1.0)
        images = [
            small.copy(),
            linear_to_bgr(self._simulate(linear)),
            linear_to_bgr(recoloured),
            linear_to_bgr(self._simulate(recoloured)),
        ]
        tiles = zip(images, tile_titles(self.viewer))
        out = np.empty_like(frame)
        cells = [(0, 0, w1, h1), (w1, 0, w - w1, h1), (0, h1, w1, h - h1), (w1, h1, w - w1, h - h1)]
        for (tile, text), (x, y, tw, th) in zip(tiles, cells):
            if tile.shape[:2] != (th, tw):
                tile = cv2.resize(tile, (tw, th), interpolation=cv2.INTER_LINEAR)
            _label(tile, text)
            out[y:y + th, x:x + tw] = tile
        return out
