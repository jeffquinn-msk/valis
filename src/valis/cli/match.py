"""``valis-match``: quick keypoint detection and matching on two images.

The command-line counterpart of the web app's "Run Keypoint Detection": both
images are downsampled to a working resolution, the moving image is optionally
flipped/rotated, each image is preprocessed, and the two are matched with the
chosen detector/matcher. It reports how many matches survive the outlier filter
and can save the preprocessed images, a match visualization and the matched
coordinates. No registration is run, so it takes seconds; use it to pick
preprocessing settings before ``valis-align``.
"""

import argparse
import csv
import json
import os
import sys

# Import valis (via valis.cli.common) before any torch-related import to avoid
# the exit-139 segfault.
from valis.cli import common


def get_parser():
    parser = argparse.ArgumentParser(
        prog="valis-match",
        description=__doc__.split("\n\n", 1)[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image", help="Moving image (e.g. .ome.tif). Only page 0 is used."
    )
    parser.add_argument(
        "--reference", help="Reference image (e.g. .ome.tif). Only page 0 is used."
    )
    parser.add_argument(
        "--output-dir",
        help="If given, write matches.png, the preprocessed images, matches.csv "
        "and match.json here.",
    )
    common.add_list_processors_arg(parser)
    common.add_preprocessing_args(parser)
    common.add_geometry_args(parser)
    common.add_size_args(parser)
    common.add_matcher_args(parser)
    return parser


def write_outputs(out_dir, img_proc, ref_proc, kp1, kp2, summary):
    """Write the preprocessed images, a match visualization, the matched
    coordinates and a JSON summary to ``out_dir``. Returns the written paths."""
    from valis import viz, warp_tools

    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "image": os.path.join(out_dir, "image_processed.png"),
        "reference": os.path.join(out_dir, "reference_processed.png"),
        "matches": os.path.join(out_dir, "matches.png"),
        "csv": os.path.join(out_dir, "matches.csv"),
        "json": os.path.join(out_dir, "match.json"),
    }
    warp_tools.save_img(paths["image"], img_proc)
    warp_tools.save_img(paths["reference"], ref_proc)
    warp_tools.save_img(
        paths["matches"],
        viz.draw_matches(
            src_img=img_proc,
            kp1_xy=kp1,
            dst_img=ref_proc,
            kp2_xy=kp2,
            rad=3,
            alignment="horizontal",
        ),
    )
    with open(paths["csv"], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_x", "image_y", "reference_x", "reference_y"])
        for (x1, y1), (x2, y2) in zip(kp1, kp2):
            w.writerow([f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}"])
    with open(paths["json"], "w") as f:
        json.dump(summary, f, indent=2)
    return paths


def main(argv=None):
    parser = get_parser()
    args = parser.parse_args(argv)
    if args.list_processors:
        common.print_processors()
        return
    if not args.image or not args.reference:
        parser.error("--image and --reference are required")
    for flag in ("image", "reference"):
        if not os.path.isfile(getattr(args, flag)):
            parser.error(f"--{flag}: not a file: {getattr(args, flag)}")

    settings = common.side_settings(parser, args)
    geometry = common.geometry_from_args(args)
    matcher_cfg = common.matcher_cfg_from_args(args, always=True)

    procs, summary = {}, {"matcher": matcher_cfg}
    for side in common.SIDES:
        path = getattr(args, side)
        processor, params = settings[side]
        processor = common.resolve_processor(processor, path)
        size = common.size_for(args, side)
        side_geometry = geometry if side == "image" else None
        print(f"[{side}] {path}: {processor} {params or ''} at {size}px", flush=True)
        procs[side] = common.preprocess(path, processor, params, size, side_geometry)
        summary[side] = {
            "path": os.path.abspath(path),
            "processor": processor,
            "params": params,
            "size": size,
            "processed_shape": list(procs[side].shape[:2]),
        }
        if side_geometry is not None:
            summary[side]["geometry"] = side_geometry

    print(f"[match] {matcher_cfg}", flush=True)
    kp1, kp2, n_total, n_filtered = common.detect_and_match(
        procs["image"], procs["reference"], matcher_cfg
    )
    summary.update(n_total=n_total, n_filtered=n_filtered)
    print(f"matches: {n_filtered} kept / {n_total} total", flush=True)

    if args.output_dir:
        paths = write_outputs(
            args.output_dir, procs["image"], procs["reference"], kp1, kp2, summary
        )
        for p in paths.values():
            print(f"wrote {p}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
