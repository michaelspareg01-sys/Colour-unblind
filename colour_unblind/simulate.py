"""Simulate what a colour-blind viewer (or a cat) sees.

The models come from the daltonlens package (MIT), following its own advice
(https://daltonlens.org/opensource-cvd-simulation/):

- Protanopia / deuteranopia (severity 1): Viénot, Brettel & Mollon 1999,
  "Digital video colourmaps for checking the legibility of displays by
  dichromats", Color Research & Application 24(4).
- Partial protan / deutan severity: Machado, Oliveira & Fernandes
  2009, "A physiologically-based model for simulation of color vision
  deficiency", IEEE TVCG 15(6). Tritan uses Brettel at every severity
  because Machado is weak for tritanopia.
- Tritan: Brettel, Viénot & Mollon 1997, "Computerized simulation of color
  appearance for dichromats", JOSA A 14(10).
- Cat: modelled as a human deuteranope. Clark & Clark 2016, "Neutral point
  testing of color vision in the domestic cat", Exp Eye Res 153:23-26, found
  the cat's neutral point near 505 nm, "nearly identical to the neutral point
  of the human deuteranope".
- Achromat: relative luminance Y only (BT.709 / sRGB weights).

All of these are linear, or two-piece linear (Brettel), in linear RGB. We
probe daltonlens once to recover the exact matrices, then apply them with
float32 numpy so video runs in real time.
"""

from dataclasses import dataclass

import numpy as np
from daltonlens import simulate as dl

from colour_unblind.colourspace import BT709_LUMA, apply_matrix

DICHROMATS = {
    "protan": dl.Deficiency.PROTAN,
    "deutan": dl.Deficiency.DEUTAN,
    "tritan": dl.Deficiency.TRITAN,
}
VIEWERS = ("protan", "deutan", "tritan", "cat", "achromat")


@dataclass(frozen=True)
class Simulation:
    """``m1 @ rgb``, or for two-piece models ``m2 @ rgb`` where ``rgb . normal < 0``."""

    m1: np.ndarray
    m2: np.ndarray | None = None
    normal: np.ndarray | None = None

    def apply(self, img: np.ndarray) -> np.ndarray:
        out = apply_matrix(img, self.m1)
        if self.m2 is None:
            return out
        other_side = (img.astype(np.float32, copy=False) @ self.normal.astype(np.float32)) < 0
        return np.where(other_side[..., None], apply_matrix(img, self.m2), out)


def _probe_linear(sim: dl.Simulator, deficiency, severity: float) -> np.ndarray:
    basis = np.eye(3).reshape(1, 3, 3)
    return sim._simulate_cvd_linear_rgb(basis, deficiency, severity)[0].T


def _probe_brettel(severity: float) -> Simulation:
    sim = dl.Simulator_Brettel1997()
    lms_from_rgb = sim.color_model.LMS_from_linearRGB
    # Same separation plane daltonlens uses (white as the neutral axis).
    lms_white = lms_from_rgb @ np.ones(3)
    n_lms = np.cross(lms_white, dl.lms_confusion_axis(dl.Deficiency.TRITAN))
    normal = lms_from_rgb.T @ n_lms

    probes = np.random.default_rng(0).random((4096, 3))
    out = sim._simulate_cvd_linear_rgb(probes.reshape(1, -1, 3), dl.Deficiency.TRITAN, severity)[0]
    side2 = probes @ normal < 0
    m1 = np.linalg.lstsq(probes[~side2], out[~side2], rcond=None)[0].T
    m2 = np.linalg.lstsq(probes[side2], out[side2], rcond=None)[0].T
    return Simulation(m1, m2, normal)


def simulation_for(viewer: str, severity: float = 1.0) -> Simulation:
    """The simulation for ``viewer`` (see ``VIEWERS``). ``severity`` in [0, 1]
    applies to protan, deutan and tritan; cats and achromats are always full."""
    if viewer not in VIEWERS:
        raise ValueError(f"unknown viewer {viewer!r}; choose from {', '.join(VIEWERS)}")
    if not 0.0 <= severity <= 1.0:
        raise ValueError(f"severity must be between 0 and 1, got {severity}")

    if viewer == "achromat":
        return Simulation(np.tile(BT709_LUMA.astype(np.float64), (3, 1)))
    if viewer == "cat":
        viewer, severity = "deutan", 1.0
    if viewer == "tritan":
        return _probe_brettel(severity)
    model = dl.Simulator_Vienot1999() if severity >= 0.999 else dl.Simulator_Machado2009()
    return Simulation(_probe_linear(model, DICHROMATS[viewer], severity))
