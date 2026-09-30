"""CLI to align two whole-slide images.

The heavy lifting now lives in :mod:`valis.interactive.pipeline` (shared with the
interactive web app). This module is a thin argparse + logging wrapper around
:func:`valis.interactive.pipeline.run_alignment`.
"""

import argparse
import logging
import os
import sys

# Import valis (and the interactive package, which imports valis) before any
# torch-related import to avoid the exit-139 segfault.
from valis.interactive import pipeline, processors

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(description="Align two WSIs.")
    parser.add_argument(
        "--reference",
        type=str,
        help="Path to WSI in .ome.tif format. Will be used as the reference image.",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to WSI in .ome.tif format. Will be used as the warped image.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory.",
    )
    parser.add_argument(
        "--max-processed-image-dim-px",
        type=int,
        default=2048,
        help="Max side length used for feature detection / non-rigid registration. "
        "Higher = better matches but more memory; 4096 OOMs on a 32GB laptop.",
    )
    parser.add_argument(
        "--image-stain",
        choices=processors.STAIN_CHOICES,
        default="auto",
        help="Preprocessor for --image. 'he-hematoxylin' deconvolves H&E and "
        "keeps the hematoxylin (nuclei) channel, automatically falling back "
        "to the sparse-dots variant if the matcher gets too few matches. "
        "'he-hematoxylin-sparse' forces the sparse path. 'inverted-"
        "fluorescence' un-inverts a DAPI-style image so nuclei are bright. "
        "'auto' lets the script decide.",
    )
    parser.add_argument(
        "--reference-stain",
        choices=processors.STAIN_CHOICES,
        default="auto",
        help="Same as --image-stain but for --reference.",
    )
    parser.add_argument(
        "--orientation-margin",
        type=float,
        default=0.0,
        help="Minimum NCC margin (best - identity) required to apply a D4 "
        "pre-rotation. Below this, the script falls back to identity. "
        "Default 0 trusts the winning transform; raise it (e.g. 0.05) only "
        "when the script's heuristic is producing wrong flips.",
    )
    parser.add_argument(
        "--no-script-orientation",
        action="store_true",
        help="Skip the script's own D4 pre-rotation entirely and let valis "
        "handle reflections (via check_for_reflections=True). Useful when the "
        "script's NCC-based orientation check is too noisy on weak/ambiguous "
        "stain pairs.",
    )
    parser.add_argument(
        "--min-rigid-matches",
        type=int,
        default=30,
        help="Minimum number of initial keypoint matches required before "
        "valis's rematch step. Below this, the script aborts with a clear "
        "error instead of letting valis warp the image with a degenerate "
        "transform (which OOMs the rematch's feature detection).",
    )

    # Detector/matcher controls, mirroring the web app's. Left unset, valis's
    # default matcher is used; setting any of them builds a LightGlue, LoMa or
    # RoMa v2 matcher with the rest taken from processors.MATCHER_SCHEMA defaults.
    schema = processors.MATCHER_SCHEMA
    parser.add_argument(
        "--matcher",
        choices=schema["matcher"]["options"],
        help=f"Feature matcher (default: {schema['matcher']['default']}). "
        "loma-b uses its own DaD + DeDoDe-G features and ignores --detector. "
        "romav2 matches densely: it ignores --detector and samples "
        "--max-keypoints matches.",
    )
    parser.add_argument(
        "--detector",
        choices=schema["detector"]["options"],
        help=f"Feature detector (default: {schema['detector']['default']}).",
    )
    parser.add_argument(
        "--max-keypoints",
        type=int,
        help="Maximum keypoints per image "
        f"(default: {schema['max_keypoints']['default']}).",
    )
    parser.add_argument(
        "--ransac-thresh",
        type=float,
        help="Match-filter reprojection threshold in pixels "
        f"(default: {schema['ransac_thresh']['default']}).",
    )
    parser.add_argument(
        "--filter-method",
        # "none" only changes what the web preview displays; not meaningful here.
        choices=[o for o in schema["filter_method"]["options"] if o != "none"],
        help="Match filter "
        f"(default: {schema['filter_method']['default']}).",
    )
    return parser


def matcher_cfg_from_args(args):
    """Return ``run_alignment`` ``matcher_cfg`` from the CLI flags, or ``None``
    when none were given (keep valis's default matcher)."""
    flags = {
        "matcher": args.matcher,
        "detector": args.detector,
        "max_keypoints": args.max_keypoints,
        "ransac_thresh": args.ransac_thresh,
        "filter_method": args.filter_method,
    }
    if all(v is None for v in flags.values()):
        return None
    return {
        k: v if v is not None else processors.MATCHER_SCHEMA[k]["default"]
        for k, v in flags.items()
    }


def setup_valis_logging():
    """Configure detailed stream logging for the valis logger."""
    valis_logger = logging.getLogger("valis")
    valis_logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d "
        "- %(funcName)s() - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    valis_logger.addHandler(console_handler)
    return valis_logger


def main():
    setup_valis_logging()
    args = get_parser().parse_args()

    print("=== align_two_images invocation ===", flush=True)
    print(f"  argv         : {' '.join(sys.argv)}", flush=True)
    print(f"  cwd          : {os.getcwd()}", flush=True)
    for k, v in sorted(vars(args).items()):
        print(f"  {k:<24}: {v}", flush=True)
    print("===================================", flush=True)

    try:
        pipeline.run_alignment(
            image_path=args.image,
            reference_path=args.reference,
            output_dir=args.output_dir,
            image_stain=args.image_stain,
            reference_stain=args.reference_stain,
            max_processed_image_dim_px=args.max_processed_image_dim_px,
            min_rigid_matches=args.min_rigid_matches,
            orientation_margin=args.orientation_margin,
            no_script_orientation=args.no_script_orientation,
            matcher_cfg=matcher_cfg_from_args(args),
        )
    except pipeline.AlignmentError as e:
        print(f"\nALIGNMENT ABORTED: {e}", flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
