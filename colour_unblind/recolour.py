"""Recolour images so a colour-blind viewer (or a cat) can tell colours apart.

Protan, deutan and tritan use the daltonizer that ships in Android
(AOSP frameworks/native, services/surfaceflinger/Effects/Daltonizer.cpp,
Apache-2.0). In LMS cone space it works out what the viewer cannot see (the
colour minus its projection onto the viewer's plane of visible colours) and
spreads that lost signal onto the cones they still have, with weight 0.7.
AOSP's comments say those weights were "chosen arbitrarily"; ``strength``
scales them (1.0 reproduces Android, 0.0 leaves the image unchanged).

Cats get the deutan correction (see ``simulate`` for why). Achromats get
contrast-preserving greyscale (see ``decolour``).
"""

import numpy as np

from colour_unblind.colourspace import apply_matrix
from colour_unblind.decolour import ContrastPreservingGrey
from colour_unblind.simulate import VIEWERS

# AOSP Daltonizer.cpp, transposed from its column-major mat4 literals.
_RGB_TO_XYZ = np.array(
    [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]
)
_XYZ_TO_LMS = np.array(
    [[0.7328, 0.4296, -0.1624], [-0.7036, 1.6975, 0.0061], [0.0030, 0.0136, 0.9834]]
)
_SPREAD = 0.7
_MISSING_CONE = {"protan": 0, "deutan": 1, "tritan": 2}


def android_matrix(kind: str, strength: float = 1.0) -> np.ndarray:
    """The 3x3 linear-RGB correction matrix for ``kind`` (protan/deutan/tritan)."""
    k = _MISSING_CONE[kind]
    rgb2lms = _XYZ_TO_LMS @ _RGB_TO_XYZ
    lms2rgb = np.linalg.inv(rgb2lms)

    # Project along the missing cone's axis onto the plane through black,
    # white and blue (protan/deutan) or red (tritan), which stays in gamut.
    anchor = rgb2lms[:, 0] if kind == "tritan" else rgb2lms[:, 2]
    normal = np.cross(rgb2lms @ np.ones(3), anchor)
    sim = np.eye(3)
    sim[k] = -normal / normal[k]
    sim[k, k] = 0.0

    spread = np.eye(3)
    spread[:, k] = _SPREAD * strength
    spread[k, k] = 1.0

    return lms2rgb @ (sim @ rgb2lms + spread @ (rgb2lms - sim @ rgb2lms))


class MatrixRecolour:
    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix

    def apply(self, img: np.ndarray) -> np.ndarray:
        return apply_matrix(img, self.matrix)


def recolour_for(viewer: str, strength: float = 1.0):
    """An object with ``.apply(linear_rgb) -> linear_rgb`` for ``viewer``."""
    if viewer not in VIEWERS:
        raise ValueError(f"unknown viewer {viewer!r}; choose from {', '.join(VIEWERS)}")
    if viewer == "achromat":
        return ContrastPreservingGrey()
    return MatrixRecolour(android_matrix("deutan" if viewer == "cat" else viewer, strength))
