"""FlatSAM2 demo: track one dataset sequence or your own video, locally or on SLURM.

All paths/logins come from .env (copy .env.example). Examples:

    python demo.py dataset                         # DEMO_DATASET / DEMO_SEQUENCE from .env
    python demo.py dataset pot V24_3
    python demo.py video clip.mp4 --corners x1 y1 x2 y2 x3 y3 x4 y4
    python demo.py dataset pot V24_3 --slurm       # GPU node: sbatch here, else ssh HPC_HOST
    python demo.py params                          # list every tunable key + current value
    python demo.py dataset pot V24_3 --set loma.inlier_threshold=40 --set symmetry.match_tau_px=20
"""

import argparse
import ast
import hashlib
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

from flatsam2.path_config import REPO_ROOT, paths  # loads .env before torch / HF imports


def run(cmd: list[str]) -> None:
    print("+", shlex.join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def flatten(d: dict, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Yield ("a.b.c", value) for every leaf of a nested dict."""
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flatten(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", v


def build_config(args: argparse.Namespace):
    """Load ``--config`` and apply every ``--set key=value`` (validated by Pydantic).

    Returns:
        The resolved TrackerConfig (or LoMaTrackerConfig).

    Raises:
        ValueError: on a malformed ``--set`` or an unknown key.
    """
    from flatsam2.config import load_config_from_file

    cfg = load_config_from_file(args.config)
    if not args.set:
        return cfg
    data = cfg.model_dump()
    for item in args.set:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"--set expects key=value, got {item!r}")
        *parents, leaf = key.split(".")
        node = data
        for p in parents:
            node = node.get(p) if isinstance(node, dict) else None
        if not isinstance(node, dict) or leaf not in node:
            raise ValueError(f"Unknown config key {key!r}; see `python demo.py params`")
        try:
            node[leaf] = ast.literal_eval(raw)  # 40, 0.5, True, None, [1, 2], 'str'
        except (ValueError, SyntaxError):
            node[leaf] = raw  # bare word, e.g. method=matches
    return type(cfg).model_validate(data)


def config_file(args: argparse.Namespace) -> Path:
    """Config file for run.py: ``--config`` as-is, or a flattened copy with ``--set`` applied.

    The copy's stem ``demo_<hash>`` names the results dir, so tweaked runs never mix with
    real experiment results and identical tweaks reuse the same dir.
    """
    if not args.set:
        return args.config
    from flatsam2.reproducibility import _config_to_python

    cfg = build_config(args)
    tag = hashlib.sha1(cfg.model_dump_json().encode()).hexdigest()[:8]
    path = paths.OUTPUT_DIRECTORY / "demo_configs" / f"demo_{tag}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_config_to_python(cfg))
    print(f"config: {path}")
    return path


def demo_params(args: argparse.Namespace) -> None:
    """Print every settable key of the (resolved) config with its value."""
    for key, value in flatten(build_config(args).model_dump()):
        print(f"{key} = {value!r}")


def demo_dataset(args: argparse.Namespace) -> None:
    """Track -> evaluate -> render one sequence through the existing CLIs."""
    cfg = ["--config", str(config_file(args))]
    run([sys.executable, "run.py", args.dataset, "--sequences", args.sequence, "--save-masks", *cfg])
    run([sys.executable, "evaluate.py", args.dataset, "--sequences", args.sequence, *cfg])
    run([sys.executable, "visualize.py", args.dataset, args.sequence, "--draw-masks", *cfg])
    print(f"results: {paths.OUTPUT_DIRECTORY}\nvideo:   {paths.VIS_DIRECTORY}")


def demo_video(args: argparse.Namespace) -> None:
    """Track a planar object through an arbitrary video and write an annotated mp4."""
    import cv2
    import numpy as np
    from einops import rearrange

    from flatsam2.datasets import warp_corners
    from flatsam2.tracker import FlatSAMTracker
    from visualize import COLOR_OUTPUT, OUTPUT_LABEL, draw_legend, draw_output

    cap = cv2.VideoCapture(str(args.video))
    ok, frame = cap.read()
    if not ok:
        raise ValueError(f"Cannot read video: {args.video}")
    init_corners = rearrange(np.asarray(args.corners, dtype=float), "(n d) -> d n", d=2)  # (2, 4)

    tracker = FlatSAMTracker(build_config(args))
    tracker.init(frame, init_corners=init_corners)
    out_path = args.output or args.video.with_name(f"{args.video.stem}_tracked.mp4")
    h, w = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    corners = init_corners
    while ok:
        draw_output(frame, corners)
        draw_legend(frame, [(OUTPUT_LABEL, COLOR_OUTPUT)])
        writer.write(frame)
        ok, frame = cap.read()
        if ok:
            corners = warp_corners(tracker.track(frame).homography, init_corners)
    writer.release()
    print(f"video: {out_path}")


def submit_slurm(argv: list[str]) -> None:
    """Re-run this same demo command on a GPU node via slurm/submit_cmd.py."""
    inner = [a for a in argv if a != "--slurm"]
    # one sequence ≈ 5 min: a small, short job on the fast partition starts sooner
    submit = ["python", "slurm/submit_cmd.py", "--gpus", "1",
              "--partition", os.environ.get("DEMO_SLURM_PARTITION", "amdgpufast"),
              "--time", os.environ.get("DEMO_SLURM_TIME", "00:30:00"),  # must stay <= 4 h on *fast
              "--mem", os.environ.get("DEMO_SLURM_MEM", "16G"),
              "--job-name", "flatsam2-demo", "--", "python", "demo.py", *inner]
    if shutil.which("sbatch"):
        run(submit)
        return
    # shortcut: assumes the HPC repo is already synced (make sjo_sync) and a video path exists there
    host = os.environ.get("HPC_HOST")
    if not host:
        sys.exit("no sbatch here and HPC_HOST not set in .env")
    repo = os.environ.get("HPC_REPO", "~/git/flatsam2")
    run(["ssh", host, f"cd {repo} && source .envrc && {shlex.join(submit)}"])


def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path,
                        default=Path(os.environ.get("DEMO_CONFIG",
                                                    "configs/tracker/loma_samh_matchnaming.py")),
                        help="Base tracker config (default: DEMO_CONFIG from .env).")
    common.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override one config value, repeatable; keys from `params`.")
    common.add_argument("--slurm", action="store_true", help="Run on a SLURM GPU node.")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="\n".join(__doc__.splitlines()[2:]))
    sub = parser.add_subparsers(dest="mode", required=True)

    ds = sub.add_parser("dataset", parents=[common],
                        help="Track one sequence of POT-210 / PlanarTrack.")
    ds.add_argument("dataset", nargs="?", default=os.environ.get("DEMO_DATASET", "pot"),
                    choices=["pot", "planartrack"])
    ds.add_argument("sequence", nargs="?", default=os.environ.get("DEMO_SEQUENCE", "V01_7"))

    vid = sub.add_parser("video", parents=[common], help="Track an object in your own video.")
    vid.add_argument("video", type=Path)
    vid.add_argument("--corners", type=float, nargs=8, metavar="XY", required=True,
                     help="Object corners in the first frame, pixels, clockwise from top-left: "
                          "x1 y1 x2 y2 x3 y3 x4 y4.")
    vid.add_argument("--output", type=Path, help="Output mp4 (default <video>_tracked.mp4).")

    sub.add_parser("params", parents=[common], help="List tunable config keys and values.")

    args = parser.parse_args()
    if args.slurm and args.mode != "params":
        submit_slurm(sys.argv[1:])
    elif args.mode == "dataset":
        demo_dataset(args)
    elif args.mode == "video":
        demo_video(args)
    else:
        demo_params(args)


if __name__ == "__main__":
    main()
