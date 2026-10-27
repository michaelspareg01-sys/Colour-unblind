"""sRGB <-> linear RGB conversion and small per-pixel helpers.

The sRGB transfer function is the one in IEC 61966-2-1:1999. Frames come in
and go out as OpenCV-style BGR uint8; everything in between is linear RGB
float32 in [0, 1], which is the space all colour vision models here work in.
"""

import numpy as np

# Rec. ITU-R BT.709 / sRGB primaries, D65 white: relative luminance Y.
BT709_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

_ENCODE_STEPS = 65535


def srgb_decode(v: np.ndarray) -> np.ndarray:
    """sRGB-encoded values in [0, 1] -> linear."""
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def srgb_encode(v: np.ndarray) -> np.ndarray:
    """Linear values in [0, 1] -> sRGB-encoded."""
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055)


_DECODE_LUT = srgb_decode(np.arange(256) / 255.0).astype(np.float32)
_ENCODE_LUT = np.round(
    srgb_encode(np.linspace(0.0, 1.0, _ENCODE_STEPS + 1)) * 255.0
).astype(np.uint8)


def bgr_to_linear(frame: np.ndarray) -> np.ndarray:
    """BGR uint8 frame -> linear RGB float32."""
    return _DECODE_LUT[frame[..., ::-1]]


def srgb8_to_linear(values: np.ndarray) -> np.ndarray:
    """sRGB-encoded uint8 values of any shape -> linear float32."""
    return _DECODE_LUT[values]


def linear_to_srgb8(values: np.ndarray) -> np.ndarray:
    """Linear float values of any shape -> sRGB-encoded uint8, clipped."""
    clipped = np.clip(np.nan_to_num(values, nan=0.0), 0.0, 1.0)
    return _ENCODE_LUT[(clipped * _ENCODE_STEPS + 0.5).astype(np.uint16)]


def linear_to_bgr(img: np.ndarray) -> np.ndarray:
    """Linear RGB float -> BGR uint8 frame. Out-of-gamut values are clipped."""
    return np.ascontiguousarray(linear_to_srgb8(img)[..., ::-1])


def apply_matrix(img: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Multiply every RGB pixel by the 3x3 matrix ``m`` (out = m @ rgb)."""
    return img.astype(np.float32, copy=False) @ np.asarray(m, dtype=np.float32).T


def luminance(img: np.ndarray) -> np.ndarray:
    """Relative luminance Y of a linear RGB image, shape (H, W)."""
    return img.astype(np.float32, copy=False) @ BT709_LUMA
