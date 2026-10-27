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

from colour_unblind.colourspace import luminance, srgb_decode, srgb_encode


def _features(rgb: np.ndarray) -> np.ndarray:
    """Constant + the 9 monomials r, g, b, r^2, g^2, b^2, rg, rb, gb."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return np.stack([np.ones_like(r), r, g, b, r * r, g * g, b * b, r * g, r * b, g * b], axis=-1)


class ContrastPreservingGrey:
    """Stateful, so one instance should handle one video stream."""

    def __init__(self, analysis_width: int = 256, smoothing: float = 0.1):
        if not 0.0 < smoothing <= 1.0:
            raise ValueError(f"smoothing must be in (0, 1], got {smoothing}")
        self.analysis_width = analysis_width
        self.smoothing = smoothing
        self.coefficients: np.ndarray | None = None

    def _fit(self, encoded: np.ndarray) -> np.ndarray | None:
        h, w = encoded.shape[:2]
        if w > self.analysis_width:
            size = (self.analysis_width, max(1, round(h * self.analysis_width / w)))
            encoded = cv2.resize(encoded, size, interpolation=cv2.INTER_AREA)
        if float(encoded.reshape(-1, 3).std(axis=0).max()) < 1e-3:
            return None  # a flat frame has no contrast to preserve
        # cv2.decolor returns an all-zero image when a channel hits exactly
        # 0 or 255 (found by testing OpenCV 5.0), so keep it just inside.
        bgr = np.clip(np.round(encoded[..., ::-1] * 255), 1, 254).astype(np.uint8)
        grey, _ = cv2.decolor(bgr)
        if grey.max() == grey.min():
            return None
        x = _features(bgr[..., ::-1].astype(np.float32) / 255).reshape(-1, 10)
        coef = np.linalg.lstsq(x, grey.reshape(-1).astype(np.float32) / 255, rcond=None)[0]
        return coef if np.all(np.isfinite(coef)) else None

    def apply(self, img: np.ndarray) -> np.ndarray:
        encoded = srgb_encode(np.clip(img, 0.0, 1.0)).astype(np.float32)
        fresh = self._fit(encoded)
        if fresh is not None:
            if self.coefficients is None:
                self.coefficients = fresh
            else:
                self.coefficients = (1 - self.smoothing) * self.coefficients + self.smoothing * fresh

        if self.coefficients is None:
            grey = luminance(img)
        else:
            grey = srgb_decode(np.clip(_features(encoded) @ self.coefficients.astype(np.float32), 0, 1))
        return np.repeat(grey.astype(np.float32)[..., None], 3, axis=-1)
