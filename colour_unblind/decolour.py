"""Greyscale that keeps colour differences visible, for viewers with no colour vision.

Plain luminance turns a red and a green of equal brightness into the same
grey. Lu, Xu & Jia 2012, "Contrast preserving decolorization" (ICCP), instead
choose a greyscale mapping that keeps colour contrast. OpenCV implements it
as ``cv2.decolor``: a degree-2 polynomial in R, G, B (9 terms) followed by a
min-max stretch.

Used on its own it has two problems for video, found by testing: it costs
~0.5 s per 720p frame, and it ignores absolute brightness (every frame is
stretched to full black..white, so night scenes turn mid-grey, and on busy
scenes white can come out darker than black). So, our additions:

- run it on a small copy of every few frames and fit its polynomial by
  least squares, so it can be applied quickly at full size;
- output = ordinary luminance + only the part of decolor's grey that
  luminance does not already explain (the colour-only contrast), scaled to
  at most the scene's own brightness spread (but at least MIN_SPREAD);
- blend each new fit into a running average to avoid flicker.
"""

import cv2
import numpy as np

from colour_unblind.colourspace import linear_to_srgb8, luminance, srgb8_to_linear

# Smallest spread (sRGB-encoded, 0-1) the colour-only contrast may have. Our choice.
MIN_SPREAD = 0.12


def _features(rgb: np.ndarray) -> np.ndarray:
    """Constant + the 9 monomials r, g, b, r^2, g^2, b^2, rg, rb, gb."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return np.stack([np.ones_like(r), r, g, b, r * r, g * g, b * b, r * g, r * b, g * b], axis=-1)


def _evaluate(rgb: np.ndarray, c: np.ndarray) -> np.ndarray:
    """``_features(rgb) @ c``, factored to avoid building the feature stack."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return c[0] + r * (c[1] + c[4] * r + c[7] * g + c[8] * b) + g * (c[2] + c[5] * g + c[9] * b) + b * (c[3] + c[6] * b)


def _encoded_luma(bgr: np.ndarray) -> np.ndarray:
    """sRGB-encoded relative luminance, 0-1, of a BGR uint8 image."""
    return linear_to_srgb8(luminance(srgb8_to_linear(bgr[..., ::-1]))).astype(np.float32) / 255


def fit_decolor(bgr: np.ndarray) -> np.ndarray | None:
    """Coefficients of the polynomial (see ``_features``) that reproduces
    ``cv2.decolor`` on this BGR uint8 image, or None if it gives no answer."""
    if float(bgr.reshape(-1, 3).std(axis=0).max()) < 0.5:
        return None  # a flat frame has no contrast to preserve
    # cv2.decolor returns an all-zero image when a channel hits exactly
    # 0 or 255, or when the image is a few flat colours (found by testing
    # OpenCV 5.0). Keep values just inside, and give up on a flat answer.
    bgr = np.clip(bgr, 1, 254)
    grey, _ = cv2.decolor(bgr)
    if grey.max() == grey.min():
        return None
    x = _features(bgr[..., ::-1].astype(np.float32) / 255).reshape(-1, 10)
    coef = np.linalg.lstsq(x, grey.reshape(-1).astype(np.float32) / 255, rcond=None)[0]
    return coef.astype(np.float32) if np.all(np.isfinite(coef)) else None


class ContrastPreservingGrey:
    """Stateful, so one instance should handle one video stream. Refits every
    ``refit_every`` frames and blends each fit into the running one with weight
    ``smoothing`` (both our choices, to avoid flicker and save time)."""

    def __init__(self, analysis_width: int = 160, smoothing: float = 0.1, refit_every: int = 3):
        if not 0.0 < smoothing <= 1.0:
            raise ValueError(f"smoothing must be in (0, 1], got {smoothing}")
        self.analysis_width = analysis_width
        self.smoothing = smoothing
        self.refit_every = max(1, refit_every)
        # 10 polynomial coefficients, then slope and offset of the part
        # luminance explains, then the gain on what is left.
        self.params: np.ndarray | None = None
        self._frames = 0

    def _fit(self, bgr: np.ndarray) -> np.ndarray | None:
        h, w = bgr.shape[:2]
        if w > self.analysis_width:
            size = (self.analysis_width, max(1, round(h * self.analysis_width / w)))
            bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
        coef = fit_decolor(bgr)
        if coef is None:
            return None
        grey = _evaluate(bgr[..., ::-1].astype(np.float32) / 255, coef).reshape(-1)
        luma = _encoded_luma(bgr).reshape(-1)
        slope, offset = np.polyfit(luma, grey, 1) if luma.std() > 1e-6 else (0.0, grey.mean())
        residual = grey - (slope * luma + offset)
        spread = residual.std()
        gain = min(1.0, max(float(luma.std()), MIN_SPREAD) / spread) if spread > 1e-6 else 0.0
        return np.concatenate([coef, [slope, offset, gain]]).astype(np.float32)

    def apply(self, img: np.ndarray) -> np.ndarray:
        bgr = linear_to_srgb8(img)[..., ::-1]
        if self._frames % self.refit_every == 0:
            fresh = self._fit(bgr)
            if fresh is not None:
                if self.params is None:
                    self.params = fresh
                else:
                    self.params = (1 - self.smoothing) * self.params + self.smoothing * fresh
        self._frames += 1

        if self.params is None:
            grey = luminance(img)
        else:
            coef, (slope, offset, gain) = self.params[:10], self.params[10:]
            luma = _encoded_luma(bgr)
            colour_only = _evaluate(bgr[..., ::-1].astype(np.float32) * np.float32(1 / 255), coef) - (slope * luma + offset)
            encoded = luma + gain * colour_only
            grey = srgb8_to_linear(np.clip(encoded * 255 + 0.5, 0, 255).astype(np.uint8))
        return np.repeat(grey[..., None], 3, axis=-1)
