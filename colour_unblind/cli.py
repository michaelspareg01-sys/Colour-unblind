"""Command line: ``colour-unblind MODE INPUT`` or ``colour-unblind live``."""

import argparse
import sys
from pathlib import Path

from colour_unblind.processor import MODES, VIEWER_NAMES, VIEWERS, FrameProcessor

MODE_HELP = {
    "simulate": "show what the viewer sees",
    "recolour": "shift colours so the viewer can tell them apart",
    "compare": "2x2 grid: original, simulated, recoloured, recoloured as they see it",
}


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-v", "--viewer", choices=VIEWERS, default="deutan",
                   help="who is watching: " + ", ".join(f"{k} = {v}" for k, v in VIEWER_NAMES.items())
                   + " (default: deutan, the most common)")
    p.add_argument("--severity", type=float, default=1.0,
                   help="0-1, how strong the protan/deutan/tritan deficiency is in simulations (default 1)")
    p.add_argument("--strength", type=float, default=1.0,
                   help="recolouring strength; 1 matches Android's built-in correction (default 1)")
    p.add_argument("--blur", type=float, default=0.0,
                   help="blur simulations by this Gaussian sigma in pixels, e.g. for cats' "
                        "lower acuity (default 0, off)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="colour-unblind",
        description="Recolour or simulate video for colour-blind people and cats.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for mode in MODES:
        p = sub.add_parser(mode, help=MODE_HELP[mode], description=MODE_HELP[mode])
        p.add_argument("input", type=Path, help="video file to process")
        p.add_argument("-o", "--output", type=Path,
                       help="output file (default: INPUT_MODE_VIEWER.mp4 next to the input)")
        p.add_argument("-q", "--quiet", action="store_true", help="no progress output")
        _add_common(p)

    live = sub.add_parser("live", help="process your webcam live (keys: m mode, v viewer, q quit)",
                          description="Process a webcam live. Keys: m = next mode, "
                                      "v = next viewer, q or Esc = quit.")
    live.add_argument("-m", "--mode", choices=MODES, default="compare", help="starting mode (default compare)")
    live.add_argument("-c", "--camera", type=int, default=0, help="camera index (default 0)")
    _add_common(live)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        mode = args.mode if args.command == "live" else args.command
        processor = FrameProcessor(args.viewer, mode, args.severity, args.strength, args.blur)
        if args.command == "live":
            from colour_unblind.live import open_camera, run_live
            run_live(processor, open_camera(args.camera))
            return 0

        from colour_unblind.video import process_video
        output = args.output or args.input.with_name(f"{args.input.stem}_{mode}_{args.viewer}.mp4")
        process_video(args.input, output, processor, progress=not args.quiet)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
