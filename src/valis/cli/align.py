"""``valis-align``: align a moving image to a reference image.

Registers --image to --reference (rigid + non-rigid) and writes
``aligned.ome.tif`` to --output-dir: page 0 is the moving image warped onto the
reference, page 1 is the reference. The pipeline is
:func:`valis.interactive.pipeline.run_alignment`, the same one the web app's
"Run Alignment" button runs.

With --preview-match it follows the web app exactly: both images are matched at
the working resolution first (as ``valis-match`` does), those matches
pre-align the moving image, and valis then refines it. Without it, the
script's own orientation check and valis's matcher do the coarse alignment.
"""

import argparse
import os
import sys

# Import valis (via valis.cli.common) before any torch-related import to avoid
# the exit-139 segfault.
from valis.cli import common
from valis.interactive import pipeline, processors


def get_parser():
    parser = argparse.ArgumentParser(
        prog="valis-align",
        description=__doc__.split("\n\n", 1)[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image",
        help="Moving image (e.g. .ome.tif). Only page 0 is used.",
    )
    parser.add_argument(
        "--reference",
        help="Reference image (e.g. .ome.tif). Only page 0 is used. Defines the "
        "output frame.",
    )
    parser.add_argument("--output-dir", help="Output directory.")
    common.add_list_processors_arg(parser)

    reg = parser.add_argument_group("registration")
    reg.add_argument(
        "--max-processed-image-dim-px",
        type=int,
        default=2048,
        help="Max side length used for feature detection / non-rigid "
        "registration. Higher = better matches but more memory; 4096 OOMs on a "
        "32GB laptop (default: 2048).",
    )
    reg.add_argument(
        "--min-rigid-matches",
        type=int,
        default=processors.ALIGNMENT_SCHEMA["min_matches"]["default"],
        help="Minimum number of keypoint matches required. Below this, abort "
        "with a clear error instead of warping with a degenerate transform. "
        "With --preview-match it applies to the preview's matches (default: "
        "%(default)s).",
    )
    reg.add_argument(
        "--orientation-margin",
        type=float,
        default=0.0,
        help="Minimum NCC margin (best - identity) required to apply a D4 "
        "pre-rotation. Below this, fall back to identity. Default 0 trusts the "
        "winning transform; raise it (e.g. 0.05) only when the orientation "
        "check produces wrong flips.",
    )
    reg.add_argument(
        "--no-script-orientation",
        action="store_true",
        help="Skip the D4 orientation pre-check and let valis handle "
        "reflections. Useful when the NCC-based check is too noisy on weak or "
        "ambiguous stain pairs. Implied by --preview-match, --flip-h, --flip-v "
        "and --rotate.",
    )
    reg.add_argument(
        "--preview-match",
        action="store_true",
        help="Match at the working resolution first and pre-align the moving "
        "image with those matches, like the web app.",
    )

    # --image-stain / --reference-stain predate the web app's "processor"
    # naming; keep them so existing invocations still work.
    common.add_preprocessing_args(parser, processor_flag="{side}-stain")
    common.add_geometry_args(parser)
    common.add_size_args(parser, group_title="working resolution (--preview-match)")
    common.add_matcher_args(
        parser,
        "Left unset, valis's default matcher is used (or, with --preview-match, "
        "the web app's defaults). Setting any of them builds a LightGlue, LoMa "
        "or RoMa v2 matcher with the rest taken from the defaults below.",
    )
    return parser


def preview_match_from_args(args, settings, geometry, matcher_cfg):
    """Preprocess and match both images at the working resolution, as the web
    app's keypoint preview does, and return a :class:`pipeline.PreviewMatch`
    plus the resolved processor names."""
    procs, resolved = {}, {}
    for side in common.SIDES:
        path = getattr(args, side)
        processor, params = settings[side]
        resolved[side] = common.resolve_processor(processor, path)
        size = common.size_for(args, side)
        print(f"[preview] {side}: {resolved[side]} {params or ''} at {size}px")
        procs[side] = common.preprocess(
            path,
            resolved[side],
            params,
            size,
            geometry if side == "image" else None,
        )
    kp1, kp2, n_total, n_filtered = common.detect_and_match(
        procs["image"], procs["reference"], matcher_cfg
    )
    print(f"[preview] matches: {n_filtered} kept / {n_total} total", flush=True)
    if n_filtered < args.min_rigid_matches:
        raise pipeline.AlignmentError(
            f"The keypoint preview found {n_filtered} matches; alignment needs "
            f"at least {args.min_rigid_matches} (--min-rigid-matches). Tune the "
            "preprocessing with valis-match first."
        )
    preview = pipeline.PreviewMatch(
        kp_moving=kp1,
        kp_reference=kp2,
        moving_img=procs["image"],
        reference_img=procs["reference"],
    )
    return preview, resolved


def main(argv=None):
    common.setup_valis_logging()
    parser = get_parser()
    args = parser.parse_args(argv)
    if args.list_processors:
        common.print_processors()
        return
    missing = [
        "--" + f.replace("_", "-")
        for f in ("image", "reference", "output_dir")
        if not getattr(args, f)
    ]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")
    for flag in ("image", "reference"):
        if not os.path.isfile(getattr(args, flag)):
            parser.error(f"--{flag}: not a file: {getattr(args, flag)}")

    settings = common.side_settings(parser, args)
    geometry = common.geometry_from_args(args)
    common.print_invocation("valis-align", sys.argv, args)

    try:
        preview = None
        stains = {side: settings[side][0] for side in common.SIDES}
        matcher_cfg = common.matcher_cfg_from_args(args, always=args.preview_match)
        if args.preview_match:
            preview, stains = preview_match_from_args(
                args, settings, geometry, matcher_cfg
            )
        aligned = pipeline.run_alignment(
            image_path=args.image,
            reference_path=args.reference,
            output_dir=args.output_dir,
            image_stain=stains["image"],
            reference_stain=stains["reference"],
            image_params=settings["image"][1],
            reference_params=settings["reference"][1],
            image_geometry=geometry,
            max_processed_image_dim_px=args.max_processed_image_dim_px,
            min_rigid_matches=args.min_rigid_matches,
            orientation_margin=args.orientation_margin,
            no_script_orientation=args.no_script_orientation,
            matcher_cfg=matcher_cfg,
            preview_match=preview,
        )
    except pipeline.AlignmentError as e:
        print(f"\nALIGNMENT ABORTED: {e}", flush=True)
        raise SystemExit(2)
    print(f"wrote {aligned}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
