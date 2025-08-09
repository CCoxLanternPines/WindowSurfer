from __future__ import annotations

import argparse

from systems.scripts.candle_viz import run_price_viz


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="WindowSurfer CLI")
    parser.add_argument("--mode", required=True, help="Mode to run")
    parser.add_argument("--tag", default="DOGEUSD")
    parser.add_argument("--speed", type=int, default=10, help="ms between frames")
    parser.add_argument("--frameskip", type=int, default=1, help="draw every k-th candle")
    parser.add_argument("--start", type=int, default=0, help="start index")
    parser.add_argument("--zoom-seconds", type=int, default=5, dest="zoom_seconds", help="seconds over which to ease-out zoom")
    parser.add_argument("--width", type=float, default=12.0)
    parser.add_argument("--height", type=float, default=6.0)
    parser.add_argument("--save", help="path to mp4/gif output")
    parser.add_argument("--grid", action="store_true", help="show grid")

    args = parser.parse_args(argv)

    if args.mode == "viz":
        run_price_viz(
            tag=args.tag,
            speed_ms=args.speed,
            frameskip=args.frameskip,
            start_idx=args.start,
            zoom_seconds=args.zoom_seconds,
            width=args.width,
            height=args.height,
            save_path=args.save,
            show_grid=args.grid,
        )
    else:
        parser.error("--mode must be 'viz'")


if __name__ == "__main__":
    main()
