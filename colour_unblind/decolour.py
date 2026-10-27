"""Greyscale that keeps colour differences visible, for viewers with no colour vision.

Plain luminance turns a red and a green of equal brightness into the same
grey. Lu, Xu & Jia 2012, "Contrast preserving decolorization" (ICCP), instead
choose a greyscale mapping that keeps colour contrast. OpenCV implements it
as ``cv2.decolor``: a degree-2 polynomial in R, G, B (9 terms) followed by a
min-max stretch.

``cv2.decolor`` takes ~0.5 s on a 720p frame and re-optimises every frame,
which would flicker. So we run it on a small copy of the frame, fit the same
polynomial (plus a constant for the stretch) by least squares, smooth those
coefficients over time, and apply them at full resolution.
"""

import cv2
import numpy as np

from colour_unblind.colourspace import linear_to_bgr, luminance, srgb8_to_linear


def _features(rgb: np.ndarray) -> np.ndarray:
    """Constant + the 9 monomials r, g, b, r^2, g^2, b^2, rg, rb, gb."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return np.stack([np.ones_like(r), r, g, b, r * r, g * g, b * b, r * g, r * b, g * b], axis=-1)


def _evaluate(rgb: np.ndarray, c: np.ndarray) -> np.ndarray:
    """``_features(rgb) @ c``, factored to avoid building the feature stack."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return c[0] + r * (c[1] + c[4] * r + c[7] * g + c[8] * b) + g * (c[2] + c[5] * g + c[9] * b) + b * (c[3] + c[6] * b)


class ContrastPreservingGrey:
    """Stateful, so one instance should handle one video stream. The mapping
    is refitted every ``refit_every`` frames and blended into the running one
    with weight ``smoothing`` (both our choices, to avoid flicker and save time)."""

    def __init__(self, analysis_width: int = 160, smoothing: float = 0.1, refit_every: int = 3):
        if not 0.0 < smoothing <= 1.0:
            raise ValueError(f"smoothing must be in (0, 1], got {smoothing}")
        self.analysis_width = analysis_width
        self.smoothing = smoothing
        self.refit_every = max(1, refit_every)
        self.coefficients: np.ndarray | None = None
        self._frames = 0

    def _fit(self, bgr: np.ndarray) -> np.ndarray | None:
        h, w = bgr.shape[:2]
        if w > self.analysis_width:
            size = (self.analysis_width, max(1, round(h * self.analysis_width / w)))
            bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
        if float(bgr.reshape(-1, 3).std(axis=0).max()) < 0.5:
            return None  # a flat frame has no contrast to preserve
        # cv2.decolor returns an all-zero image when a channel hits exactly
        # 0 or 255 (found by testing OpenCV 5.0), so keep it just inside.
        bgr = np.clip(bgr, 1, 254)
        grey, _ = cv2.decolor(bgr)
        if grey.max() == grey.min():
            return None
        # Our addition: cv2.decolor sometimes returns brightness the wrong way
        # round (dark things light), and which way varies between frames. Keep
        # its contrast but orient it like ordinary luminance.
        luma = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if luma.std() > 0 and np.corrcoef(grey.ravel(), luma.ravel())[0, 1] < 0:
            grey = 255 - grey
        x = _features(bgr[..., ::-1].astype(np.float32) / 255).reshape(-1, 10)
        coef = np.linalg.lstsq(x, grey.reshape(-1).astype(np.float32) / 255, rcond=None)[0]
        return coef.astype(np.float32) if np.all(np.isfinite(coef)) else None

    def apply(self, img: np.ndarray) -> np.ndarray:
        bgr = linear_to_bgr(img)
        if self._frames % self.refit_every == 0:
            fresh = self._fit(bgr)
            if fresh is not None:
                if self.coefficients is None:
                    self.coefficients = fresh
                else:
                    self.coefficients = (1 - self.smoothing) * self.coefficients + self.smoothing * fresh
        self._frames += 1

        if self.coefficients is None:
            grey = luminance(img)
        else:
            encoded = _evaluate(bgr[..., ::-1].astype(np.float32) * np.float32(1 / 255), self.coefficients)
            grey = srgb8_to_linear(np.clip(encoded * 255 + 0.5, 0, 255).astype(np.uint8))
        return np.repeat(grey[..., None], 3, axis=-1)
