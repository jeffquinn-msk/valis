"""Argument helpers shared by the ``valis-match`` and ``valis-align`` CLIs.

They expose the web app's controls as flags: a preprocessor + parameters per
image, a flip/rotate geometry for the moving image, a working resolution per
image, and the detector/matcher block. Defaults and valid values come from the
schemas in :mod:`valis.interactive.processors`, the same ones the web app
renders.

Importing this module imports valis (via ``valis.interactive``), so import it
before anything that pulls in torch.
"""

import json
import logging

import pyvips

from valis.interactive import processors

SIDES = ("image", "reference")
PROCESSOR_CHOICES = ("auto", *processors.PROCESSOR_REGISTRY)

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


# ---------------------------------------------------------------------------
# Preprocessing (per image)
# ---------------------------------------------------------------------------


def add_preprocessing_args(parser, processor_flag="{side}-processor"):
    """Add ``--<side>-processor`` / ``--<side>-param`` for both images.

    ``processor_flag`` names the processor flag, so ``valis-align`` can keep
    its existing ``--image-stain`` / ``--reference-stain`` spelling.
    """
    group = parser.add_argument_group(
        "preprocessing",
        "Preprocessor and parameters for each image. Run with --list-processors "
        "to see every processor's parameters, ranges and defaults.",
    )
    for side in SIDES:
        group.add_argument(
            "--" + processor_flag.format(side=side),
            dest=f"{side}_processor",
            choices=PROCESSOR_CHOICES,
            default="auto",
            help=f"Preprocessor for --{side}. 'auto' picks he-hematoxylin for RGB "
            "images and fluorescence / inverted-fluorescence for single-band "
            "images, based on brightness (default: auto).",
        )
        group.add_argument(
            f"--{side}-param",
            dest=f"{side}_params",
            action="append",
            default=[],
            metavar="NAME=VALUE",
            help=f"Preprocessor parameter for --{side}, e.g. sparse_pct=85. "
            "Repeat for several parameters.",
        )


def add_list_processors_arg(parser):
    parser.add_argument(
        "--list-processors",
        action="store_true",
        help="Print every preprocessor and its parameters, then exit.",
    )


def print_processors():
    """Print :data:`processors.PARAM_SCHEMA` as a readable table."""
    for name, spec in processors.PARAM_SCHEMA.items():
        print(f"{name}  (input: {spec['input']})")
        if not spec["params"]:
            print("    (no parameters)")
        for p in spec["params"]:
            if p["type"] == "enum":
                values = "one of " + ", ".join(p["options"])
            elif p["type"] == "bool":
                values = "true/false"
            else:
                values = f"{p['type']} {p['min']}..{p['max']}"
            print(f"    {p['name']:<20} {values:<36} default {p['default']}")
        print()


def parse_params(processor, items):
    """Turn ``["name=value", ...]`` into typed ``process_image`` kwargs.

    Names and types are checked against ``processors.PARAM_SCHEMA``. Values
    outside a slider's range are allowed, since the range is only the web app's
    UI limit.
    """
    if not items:
        return {}
    if processor == "auto":
        raise ValueError("set a processor explicitly to pass parameters to it")
    schema = {p["name"]: p for p in processors.PARAM_SCHEMA[processor]["params"]}
    params = {}
    for item in items:
        name, sep, raw = item.partition("=")
        name, raw = name.strip(), raw.strip()
        if not sep or not name:
            raise ValueError(f"expected NAME=VALUE, got {item!r}")
        p = schema.get(name)
        if p is None:
            valid = ", ".join(schema) or "none"
            raise ValueError(
                f"{processor} has no parameter {name!r} (parameters: {valid})"
            )
        params[name] = _convert(p, raw)
    return params


def _convert(p, raw):
    kind = p["type"]
    try:
        if kind == "bool":
            if raw.lower() in _TRUE:
                return True
            if raw.lower() in _FALSE:
                return False
            raise ValueError
        if kind == "int":
            return int(raw)
        if kind == "float":
            return float(raw)
        if kind == "enum":
            if raw not in p["options"]:
                raise ValueError
            return raw
    except ValueError:
        pass
    expected = "one of " + ", ".join(p["options"]) if kind == "enum" else kind
    raise ValueError(f"{p['name']}: expected {expected}, got {raw!r}")


def side_settings(parser, args):
    """Return ``{side: (processor, params)}``, exiting via ``parser.error`` on
    a bad ``--<side>-param``."""
    out = {}
    for side in SIDES:
        processor = getattr(args, f"{side}_processor")
        try:
            params = parse_params(processor, getattr(args, f"{side}_params"))
        except ValueError as e:
            parser.error(f"--{side}-param: {e}")
        out[side] = (processor, params)
    return out


# ---------------------------------------------------------------------------
# Geometry (moving image only)
# ---------------------------------------------------------------------------


def add_geometry_args(parser):
    group = parser.add_argument_group(
        "moving-image geometry",
        "Flip/rotate the moving image (--image) before preprocessing. Flips are "
        "applied first. The reference defines the output frame and is never "
        "transformed.",
    )
    group.add_argument("--flip-h", action="store_true", help="Flip horizontally.")
    group.add_argument("--flip-v", action="store_true", help="Flip vertically.")
    group.add_argument(
        "--rotate",
        type=int,
        choices=processors.ROTATIONS,
        default=0,
        help="Clockwise rotation in degrees (default: 0).",
    )


def geometry_from_args(args):
    return processors.normalize_geometry(
        {"flip_h": args.flip_h, "flip_v": args.flip_v, "rotate": args.rotate}
    )


# ---------------------------------------------------------------------------
# Working resolution (per image)
# ---------------------------------------------------------------------------


def add_size_args(parser, group_title="working resolution"):
    schema = processors.RESOLUTION_SCHEMA
    group = parser.add_argument_group(
        group_title,
        "Longest side, in px, of the thumbnail each image is preprocessed and "
        "matched at (never above the image's own size).",
    )
    group.add_argument(
        "--size",
        type=int,
        default=schema["default"],
        help=f"Working resolution for both images (default: {schema['default']}).",
    )
    for side in SIDES:
        group.add_argument(
            f"--{side}-size",
            type=int,
            help=f"Working resolution for --{side} only (default: --size).",
        )


def size_for(args, side):
    size = getattr(args, f"{side}_size")
    return args.size if size is None else size


# ---------------------------------------------------------------------------
# Detector / matcher
# ---------------------------------------------------------------------------


def add_matcher_args(parser, defaults_note=""):
    """Add the web app's detector/matcher controls. Every flag defaults to
    ``None`` so callers can tell whether it was set; fill unset ones with
    :func:`matcher_cfg_from_args`."""
    schema = processors.MATCHER_SCHEMA
    group = parser.add_argument_group("detector / matcher", defaults_note or None)
    group.add_argument(
        "--matcher",
        choices=schema["matcher"]["options"],
        help=f"Feature matcher (default: {schema['matcher']['default']}). "
        "loma-b uses its own DaD + DeDoDe-G features and ignores --detector. "
        "romav2 matches densely: it ignores --detector and samples "
        "--max-keypoints matches.",
    )
    group.add_argument(
        "--detector",
        choices=schema["detector"]["options"],
        help=f"Feature detector (default: {schema['detector']['default']}).",
    )
    group.add_argument(
        "--max-keypoints",
        type=int,
        help="Maximum keypoints per image "
        f"(default: {schema['max_keypoints']['default']}).",
    )
    group.add_argument(
        "--ransac-thresh",
        type=float,
        help="Match-filter reprojection threshold in pixels "
        f"(default: {schema['ransac_thresh']['default']}).",
    )
    group.add_argument(
        "--filter-method",
        choices=schema["filter_method"]["options"],
        help="Geometric outlier filter for matches; 'none' keeps every match "
        f"(default: {schema['filter_method']['default']}).",
    )


MATCHER_KEYS = (
    "matcher",
    "detector",
    "max_keypoints",
    "ransac_thresh",
    "filter_method",
)


def matcher_cfg_from_args(args, always=False):
    """Return ``build_matcher`` kwargs from the matcher flags, unset ones
    filled from ``processors.MATCHER_SCHEMA``. Returns ``None`` when no flag
    was given, unless ``always``."""
    flags = {k: getattr(args, k) for k in MATCHER_KEYS}
    if not always and all(v is None for v in flags.values()):
        return None
    return {
        k: v if v is not None else processors.MATCHER_SCHEMA[k]["default"]
        for k, v in flags.items()
    }


# ---------------------------------------------------------------------------
# Preprocessing + matching (the web app's "Run Keypoint Detection")
# ---------------------------------------------------------------------------


def resolve_processor(processor, path):
    if processor == "auto":
        return processors.resolve_auto_stain(path)
    return processor


def preprocess(path, processor, params, size, geometry=None):
    """Thumbnail ``path`` at ``size``, apply ``geometry``, then run
    ``processor`` with ``params``: what the web app's preview panel shows.
    ``processor`` must already be resolved (not ``auto``)."""
    v = pyvips.Image.new_from_file(path, page=0)
    if processors.PROCESSOR_INPUT.get(processor, "gray") == "rgb":
        thumb = processors.pyvips_to_thumbnail_rgb_array(v, int(size))
    else:
        thumb = processors.pyvips_to_thumbnail_array(v, int(size))
    if not processors.geometry_is_identity(geometry):
        thumb = processors.apply_geometry_array(thumb, geometry)
    cls, base_kw = processors.PROCESSOR_REGISTRY[processor]
    return processors.run_processor_on_thumbnail(
        [cls, {**base_kw, **(params or {})}], thumb, path
    )


def detect_and_match(img_proc, ref_proc, matcher_cfg):
    """Match two preprocessed thumbnails exactly as the web app does.

    Returns ``(kp_image, kp_reference, n_total, n_filtered)``.
    """
    # The web app's matching module is plain valis + numpy (no FastAPI).
    from valis.webapp import matching

    return matching.detect_and_match(img_proc, ref_proc, **matcher_cfg)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def setup_valis_logging(verbose=True):
    """Stream the ``valis`` logger to stderr."""
    valis_logger = logging.getLogger("valis")
    valis_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d "
            "- %(funcName)s() - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    valis_logger.addHandler(handler)
    return valis_logger


def print_invocation(name, argv, args):
    print(f"=== {name} invocation ===", flush=True)
    print(f"  argv : {' '.join(argv)}", flush=True)
    for k, v in sorted(vars(args).items()):
        print(f"  {k:<28}: {json.dumps(v) if isinstance(v, list) else v}", flush=True)
    print("=" * (len(name) + 19), flush=True)
