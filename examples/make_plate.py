"""Make an Ishihara-style test plate for one viewer.

The digit's dots differ from the background only in the colour direction
that viewer cannot see, so the "simulate" view hides the digit and the
"recolour" view should bring it back. Not a medical test.

    python examples/make_plate.py deutan plate.png
"""
import sys
import cv2
import numpy as np
from colour_unblind.colourspace import linear_to_bgr
from colour_unblind.simulate import simulation_for

def plate(viewer="deutan", size=480, seed=0, digit="5"):
    rng = np.random.default_rng(seed)
    mask = np.zeros((size, size), np.uint8)
    cv2.putText(mask, digit, (int(size * .22), int(size * .80)), cv2.FONT_HERSHEY_SIMPLEX, size / 75, 255, int(size / 14))
    sim = simulation_for(viewer)
    img = np.full((size, size, 3), 0.9, np.float32)  # linear, near-white paper
    for _ in range(2600):
        r = rng.uniform(size / 110, size / 45)
        x, y = rng.uniform(0, size, 2)
        if (x - size / 2) ** 2 + (y - size / 2) ** 2 > (size / 2 - r) ** 2:
            continue
        base = np.array([0.45, 0.32, 0.10]) * rng.uniform(0.6, 1.2)  # olive/orange family
        if viewer == "achromat":
            target = base.copy()
            if mask[int(y), int(x)]:
                target = np.array([base[0] * 1.6, base[1] * 0.55, base[2] * 2.0])
                target *= (base @ [0.2126, 0.7152, 0.0722]) / (target @ [0.2126, 0.7152, 0.0722])
        else:
            seen = sim.apply(base[None, None].astype(np.float32))[0, 0]
            # move off the viewer's plane: base + k*(base - seen) keeps what they see
            direction = base - seen
            k = 1.2 if mask[int(y), int(x)] else -0.6
            target = base + k * direction / (np.abs(direction).max() + 1e-9) * 0.25
        cv2.circle(img, (int(x), int(y)), int(r), np.clip(target, 0, 1).tolist(), -1, cv2.LINE_AA)
    return linear_to_bgr(img), mask

if __name__ == "__main__":
    viewer, out = sys.argv[1], sys.argv[2]
    bgr, mask = plate(viewer)
    cv2.imwrite(out, bgr)
    cv2.imwrite(out.replace(".png", "_mask.png"), mask)
