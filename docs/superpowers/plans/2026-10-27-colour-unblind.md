# Colour Unblind Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Python CLI that recolours or simulates video (files and webcam) for protan, deutan, tritan, cat and achromat viewers.

**Architecture:** Pure functions over linear-RGB float32 arrays, driven by matrices extracted once from `daltonlens` and AOSP's daltonizer; a `FrameProcessor` composes them per mode; thin I/O layers for files (ffmpeg pipe) and webcam (OpenCV window).

**Tech Stack:** Python >= 3.10, numpy, opencv-python, daltonlens==0.1.5, pytest; ffmpeg binary optional.

**Spec:** `docs/superpowers/specs/2026-10-27-colour-unblind-design.md`

## Global Constraints

- Frames in and out are BGR uint8 `(H, W, 3)` arrays (OpenCV convention).
- Every colour transform cites its source in a docstring.
- Viewer names: `protan`, `deutan`, `tritan`, `cat`, `achromat`.
- Mode names: `simulate`, `recolour`, `compare`.

## Review Focus

1. Odd frame sizes (e.g. 641x479) in `compare` mode: grid must still equal the input size.
2. Input video with no audio track: ffmpeg mapping must not fail (`-map 1:a?`).
3. Pure white and black stay white and black in every mode/viewer (no tint).
4. Severity 0 returns the input unchanged for protan/deutan/tritan.
5. A single-colour frame (zero chroma) through achromat recolour: polynomial fit must not blow up (fallback to luminance).

---

### Task 1: Colour space helpers

**Files:** Create `colour_unblind/__init__.py`, `colour_unblind/colourspace.py`, `pyproject.toml`; Test `tests/test_colourspace.py`

**Produces:** `bgr_to_linear(frame: uint8 HxWx3) -> float32 HxWx3 (RGB order)`, `linear_to_bgr(img: float) -> uint8`, `apply_matrix(img, m: 3x3) -> float32`, `luminance(img) -> float32 HxW`.

- [ ] Tests: round-trip all 256 grey levels exactly; `bgr_to_linear` of sRGB 128 is 0.2158605 (+-1e-5, IEC 61966-2-1); channel order swaps BGR->RGB; `luminance` of linear (1,0,0),(0,1,0),(0,0,1) is 0.2126/0.7152/0.0722.
- [ ] Watch fail, implement, pass, commit.

### Task 2: Simulation (daltonlens-backed)

**Files:** Create `colour_unblind/simulate.py`; Test `tests/test_simulate.py`

**Consumes:** Task 1. **Produces:** `simulation_for(viewer: str, severity: float = 1.0) -> Simulation` with `.apply(linear_rgb) -> linear_rgb`.

- Protan/deutan: severity >= 0.999 Viénot 1999, else Machado 2009. Tritan: Brettel 1997 (two-piece: `M1`, `M2`, separation normal in RGB). Cat: deutan, severity 1. Achromat: luminance replicated to 3 channels.
- Brettel extraction: separation normal `n_rgb = LMS_from_linearRGB.T @ cross(LMS_from_linearRGB @ [1,1,1], confusion_axis)`; fit each piece by least squares on probe colours from that side.

- [ ] Tests: for each of protan/deutan/tritan at severities 1.0 and 0.5, `apply` matches `daltonlens` `_simulate_cvd_linear_rgb` on 10k random linear colours within 1e-5; cat equals deutan; achromat output channels equal and equal `luminance`; severity 0 is identity; white stays white (+-1e-3).
- [ ] Watch fail, implement, pass, commit.

### Task 3: Recolouring

**Files:** Create `colour_unblind/recolour.py`, `colour_unblind/decolour.py`; Test `tests/test_recolour.py`, `tests/test_decolour.py`

**Produces:** `android_matrix(kind: 'protan'|'deutan'|'tritan', strength=1.0) -> 3x3`; `recolour_for(viewer, strength=1.0) -> object with .apply(linear_rgb)`; `ContrastPreservingGrey(analysis_width=256, smoothing=0.1).apply(linear_rgb) -> linear_rgb` (stateful).

- [ ] Tests: white/black unchanged; red vs green become further apart under deutan simulation after recolour (distance increases); strength scales only the 0.7 off-diagonal spread terms, so strength 0 is the identity; strength 1 matches the AOSP formula `lms2rgb * (sim*rgb2lms + err*(rgb2lms - sim*rgb2lms))` transcribed from column-major `mat4`; cat == deutan; decolour fit reproduces `cv2.decolor` on a 256-wide test image with correlation > 0.99; a red/green isoluminant pair gets different grey values (unlike CIE Y); flat single-colour frame doesn't raise and returns finite values.
- [ ] Watch fail, implement, pass, commit.

### Task 4: FrameProcessor and compare grid

**Files:** Create `colour_unblind/processor.py`; Test `tests/test_processor.py`

**Produces:** `FrameProcessor(viewer, mode, severity=1.0, strength=1.0, blur=0.0)` with `.process(bgr) -> bgr`, `with_viewer(v)`, `with_mode(m)`; constants `VIEWERS`, `MODES`.

- [ ] Tests: output shape == input shape for all modes, including odd 641x479; compare top-left tile equals downscaled original; `blur>0` only affects simulate output; invalid viewer raises ValueError.
- [ ] Watch fail, implement, pass, commit.

### Task 5: Video file I/O and CLI

**Files:** Create `colour_unblind/video.py`, `colour_unblind/cli.py`, `colour_unblind/__main__.py`; Test `tests/test_cli.py`

**Produces:** `process_video(input, output, processor, progress=True) -> int frames`; `main(argv) -> int`.

- [ ] Tests (generate clips with ffmpeg `testsrc` + `sine` audio, and one without audio): output exists, same frame count and size, audio stream present when input had one, absent and no crash when not; `--help` lists modes and viewers; unknown viewer exits non-zero.
- [ ] Watch fail, implement, pass, commit.

### Task 6: Live webcam

**Files:** Create `colour_unblind/live.py`; Test `tests/test_live.py`

**Produces:** `run_live(processor, capture, show, wait_key) -> None` with injectable capture/show/wait_key for tests; CLI `live` subcommand wiring `cv2.VideoCapture(index)`.

- [ ] Tests: fake capture of 3 frames + key sequence `m`, `v`, `q` cycles mode then viewer then stops; capture that fails to open raises a clear error.
- [ ] Watch fail, implement, pass, commit.

### Task 7: README with sources, demo

**Files:** Create `README.md`, `LICENSE` (MIT), `tests/data/` none (generated at test time).

- [ ] Install, usage, keys, science table with links, limitations. Commit.
