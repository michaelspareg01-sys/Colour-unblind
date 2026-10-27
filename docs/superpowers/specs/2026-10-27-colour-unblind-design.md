# Colour Unblind: design

## Intent

Turn a video (a file or a live webcam feed) into colours that a colour-blind
person, or a cat, can tell apart. Also show a normal-sighted person what the
video looks like to that viewer. Both modes live in one tool so you can
compare them.

## Decisions the user made (2026-10-27)

- Output: **both** recolour and simulate, switchable, plus a comparison view.
- Viewers: red-green (protan, deutan), blue-yellow (tritan), cats, total
  colour blindness (achromatopsia).
- Interface: **command-line tool** (Python).
- Sources: **video files and live webcam**.

## Goals

- `colour-unblind recolour|simulate|compare INPUT -v VIEWER -o OUT` for files.
- `colour-unblind live -v VIEWER` for the webcam, with keys to switch mode and viewer.
- Keep the audio track when writing files.
- Every colour transform comes from a published, citable source. Where we make
  our own choice (a default strength, smoothing), it is labelled as ours.
- Fast enough for live webcam at 640x480 on a laptop CPU.

## Non-goals

- A GUI or web app (user chose CLI).
- Claiming medical accuracy. Simulations are models, not diagnoses.
- Restoring colour for achromats (impossible: they have no working cones). For
  them "recolour" means "keep colour differences visible as brightness
  differences".

## Anti-goals

- Made-up colour maths with no source.
- Flicker in video output from per-frame adaptive methods.

## Science and sources

| Viewer | Simulate | Recolour |
|---|---|---|
| protan / deutan | Viénot, Brettel & Mollon 1999 at full severity; Machado, Oliveira & Fernandes 2009 for partial severity (DaltonLens's own recommendation) | Android `Daltonizer.cpp` (AOSP, Apache-2.0): LMS projection + error spread |
| tritan | Brettel, Viénot & Mollon 1997 (DaltonLens recommends it for tritan) | Android `Daltonizer.cpp` tritan branch |
| cat | Treated as a human deuteranope: Clark & Clark 2016 (Exp Eye Res 153:23-26, PMID 27720709) found the cat's neutral point near 505 nm, "nearly identical to the neutral point of the human deuteranope". Optional blur: cats resolve roughly 4-7x less detail than humans (Caves, Brandley & Johnsen 2018, TREE 33(5)) | Android deutan correction |
| achromat | CIE Y relative luminance from linear sRGB (0.2126, 0.7152, 0.0722; IEC 61966-2-1 / ITU-R BT.709) | Contrast-preserving decolorisation, Lu, Xu & Jia 2012 (ICCP), via OpenCV `cv2.decolor` |

Simulation code comes from the `daltonlens` package (MIT, v0.1.5), which
carries the Machado matrices from the authors' site via colour-science.

## Approaches considered

1. **Call `daltonlens` per frame.** Simple and exact, but 0.09-0.24 s per 720p
   frame (float64). Too slow for live use.
2. **Extract the linear maps from `daltonlens` once, apply them with float32
   numpy (chosen).** Every model above is linear (Viénot, Machado, Android) or
   two-piece linear (Brettel) in linear RGB. We recover the exact matrices by
   probing the simulator, then apply them per frame. Tests check we match
   `daltonlens` output.
3. **3D lookup tables.** Fast on GPU, but the extra interpolation error and
   complexity buy nothing over (2) on CPU.

For achromat recolour, `cv2.decolor` costs ~0.5 s per 720p frame and its
per-frame optimisation would flicker. OpenCV's implementation produces a
degree-2 polynomial in R, G, B followed by a min-max stretch, so we run it on
a downscaled frame, least-squares fit the 9 polynomial terms plus a constant,
smooth those coefficients over time (exponential moving average, our choice:
weight 0.1 for each new frame), and apply them at full resolution.

## How it works

- Pixels: BGR uint8 frames (OpenCV) -> linear RGB float32 through a 256-entry
  sRGB decode table -> transform -> clip to [0, 1] -> sRGB encode through a
  4096-entry table -> BGR uint8.
- Modes:
  - `simulate`: what the viewer sees.
  - `recolour`: Android daltonizer output. `--strength` scales Android's 0.7
    spread coefficients (our parameter; 1.0 reproduces Android exactly).
  - `compare`: a 2x2 grid at the input size: Original, Simulated, Recoloured,
    Recoloured as the viewer sees it. Each tile labelled.
- `--severity 0..1` for protan/deutan/tritan simulation (default 1.0).
- Files: read with OpenCV; write by piping raw frames into `ffmpeg` (H.264,
  yuv420p), copying the input's audio (re-encoded to AAC). If `ffmpeg` is
  missing, fall back to OpenCV's writer with no audio and print a warning.
- Live: OpenCV window. Keys: `m` cycle mode, `v` cycle viewer, `q`/Esc quit.

## Calls left to the builder (marked as Claude's calls)

- Package name `colour_unblind`, console script `colour-unblind`.
- EMA weight 0.1 and a 256-pixel-wide analysis frame for decolorisation.
- Default blur 0 (off); viewing distance and screen size decide what blur is
  realistic, and we don't know them.
- Rod monochromats see with rods (peak ~507 nm), so reds look darker to them
  than CIE Y predicts. We use CIE Y and say so in the README.
