import numpy as np
import pytest
from daltonlens import simulate as dl

from colour_unblind.colourspace import luminance
from colour_unblind.simulate import simulation_for

DEFICIENCY = {"protan": dl.Deficiency.PROTAN, "deutan": dl.Deficiency.DEUTAN, "tritan": dl.Deficiency.TRITAN}


def reference(viewer, severity, img):
    """What daltonlens itself produces, using the model the spec picks."""
    if viewer == "tritan":
        sim = dl.Simulator_Brettel1997()
    elif severity >= 0.999:
        sim = dl.Simulator_Vienot1999()
    else:
        sim = dl.Simulator_Machado2009()
    return sim._simulate_cvd_linear_rgb(img.astype(np.float64), DEFICIENCY[viewer], severity)


@pytest.fixture
def colours():
    rng = np.random.default_rng(0)
    return rng.random((100, 100, 3), dtype=np.float32)


@pytest.mark.parametrize("viewer", ["protan", "deutan", "tritan"])
@pytest.mark.parametrize("severity", [1.0, 0.5, 0.25])
def test_matches_daltonlens(viewer, severity, colours):
    out = simulation_for(viewer, severity).apply(colours)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, reference(viewer, severity, colours), atol=1e-5)


def test_cat_sees_like_a_deuteranope(colours):
    np.testing.assert_allclose(
        simulation_for("cat").apply(colours), simulation_for("deutan", 1.0).apply(colours)
    )


def test_achromat_sees_luminance_only(colours):
    out = simulation_for("achromat").apply(colours)
    for c in range(3):
        np.testing.assert_allclose(out[..., c], luminance(colours), atol=1e-6)


@pytest.mark.parametrize("viewer", ["protan", "deutan", "tritan"])
def test_severity_zero_is_identity(viewer, colours):
    np.testing.assert_allclose(simulation_for(viewer, 0.0).apply(colours), colours, atol=1e-6)


@pytest.mark.parametrize("viewer", ["protan", "deutan", "tritan", "cat", "achromat"])
def test_white_and_black_are_neutral(viewer):
    img = np.array([[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]], dtype=np.float32)
    np.testing.assert_allclose(simulation_for(viewer).apply(img), img, atol=1e-3)


def test_rejects_unknown_viewer_and_bad_severity():
    with pytest.raises(ValueError):
        simulation_for("dog")
    with pytest.raises(ValueError):
        simulation_for("deutan", 1.5)
