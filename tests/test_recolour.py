import numpy as np
import pytest

from colour_unblind.recolour import android_matrix, recolour_for
from colour_unblind.simulate import simulation_for


def aosp_reference(kind):
    """Row-major transcription of AOSP Daltonizer.cpp (column-major mat4 there)."""
    rgb2xyz = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz2lms = np.array([[0.7328, 0.4296, -0.1624], [-0.7036, 1.6975, 0.0061], [0.0030, 0.0136, 0.9834]])
    rgb2lms = xyz2lms @ rgb2xyz
    lms2rgb = np.linalg.inv(rgb2lms)
    w, r, b = rgb2lms @ np.ones(3), rgb2lms[:, 0], rgb2lms[:, 2]
    p0, p1 = np.cross(w, b), np.cross(w, r)
    sim = {
        "protan": np.array([[0, -p0[1] / p0[0], -p0[2] / p0[0]], [0, 1, 0], [0, 0, 1]]),
        "deutan": np.array([[1, 0, 0], [-p0[0] / p0[1], 0, -p0[2] / p0[1]], [0, 0, 1]]),
        "tritan": np.array([[1, 0, 0], [0, 1, 0], [-p1[0] / p1[2], -p1[1] / p1[2], 0]]),
    }[kind]
    err = {
        "protan": np.array([[1, 0, 0], [0.7, 1, 0], [0.7, 0, 1]]),
        "deutan": np.array([[1, 0.7, 0], [0, 1, 0], [0, 0.7, 1]]),
        "tritan": np.array([[1, 0, 0.7], [0, 1, 0.7], [0, 0, 1]]),
    }[kind]
    return lms2rgb @ (sim @ rgb2lms + err @ (rgb2lms - sim @ rgb2lms))


@pytest.mark.parametrize("kind", ["protan", "deutan", "tritan"])
def test_matches_aosp_daltonizer(kind):
    np.testing.assert_allclose(android_matrix(kind), aosp_reference(kind), atol=1e-9)


@pytest.mark.parametrize("kind", ["protan", "deutan", "tritan"])
def test_strength_zero_is_identity(kind):
    np.testing.assert_allclose(android_matrix(kind, strength=0.0), np.eye(3), atol=1e-9)


@pytest.mark.parametrize("viewer", ["protan", "deutan", "tritan", "cat"])
def test_white_and_black_unchanged(viewer):
    img = np.array([[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]], dtype=np.float32)
    np.testing.assert_allclose(recolour_for(viewer).apply(img), img, atol=1e-4)


@pytest.mark.parametrize("viewer", ["protan", "deutan", "tritan", "cat"])
@pytest.mark.parametrize("colour", [(0.8, 0.1, 0.1), (0.2, 0.6, 0.1), (0.1, 0.2, 0.8), (0.7, 0.2, 0.6)])
def test_recolouring_separates_colours_the_viewer_confuses(viewer, colour):
    sees = simulation_for(viewer)
    a = np.array([[colour]], dtype=np.float32)
    b = sees.apply(a)  # a projection: the viewer sees a and b as the same colour
    pair = np.concatenate([a, b], axis=1)
    before = np.linalg.norm(np.diff(sees.apply(pair), axis=1))
    after = np.linalg.norm(np.diff(sees.apply(np.clip(recolour_for(viewer).apply(pair), 0, 1)), axis=1))
    assert before < 1e-3
    assert after > 0.02


def test_rejects_unknown_viewer():
    with pytest.raises(ValueError):
        recolour_for("dog")
