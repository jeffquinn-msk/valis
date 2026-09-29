"""Preprocessors, thumbnail helpers, and the processor registry shared by the
``align_two_images`` CLI and the interactive web app.

This module deliberately imports ``valis`` (and nothing torch-related) at the top
so that importing it never triggers the valis-before-torch segfault.
"""

import numpy as np
import pyvips

from valis import preprocessing


class HematoxylinExtractor(preprocessing.ImageProcesser):
    """Extract the hematoxylin (nuclei) channel from an H&E image.

    Pipeline: Macenko-style normalization against a standard H&E
    reference (so faded slides come back up to consistent intensity),
    then deconvolve using ``skimage.color.rgb2hed`` with the fixed
    Ruifrok-Johnston stain matrix and keep only the H channel.

    Macenko's ``normalize_he`` picks H vs E by an angle heuristic on the
    candidate stain vectors' red component. On eosin-dominant or
    hematoxylin-faded slides that ordering can flip, sending eosin into
    the "hematoxylin" row — yielding an output where stroma lights up
    brighter than nuclei. We detect that here by checking which row
    correlates with bluish pixels in the original RGB (true hematoxylin)
    and swap if needed.

    Pass ``use_macenko=False`` to skip normalization entirely and just
    run ``rgb2hed`` on raw RGB — useful as a fallback when Macenko
    misbehaves on an atypical slide.
    """

    def create_mask(self):
        from valis.preprocessing import create_tissue_mask_from_rgb

        _, tissue_mask = create_tissue_mask_from_rgb(self.image)
        return tissue_mask

    def process_image(
        self,
        *args,
        use_macenko: bool = True,
        sparse: bool = False,
        sparse_pct: float = 90.0,
        sparse_blur_sigma: float = 1.5,
        **kwargs,
    ):
        """Extract hematoxylin channel.

        ``sparse=True`` thresholds the H channel at the ``sparse_pct``
        percentile and zeros everything below — yielding a punctate
        "nuclei dots" image that resembles a fluorescence reference.
        Use this on eosin-dominant / hematoxylin-faded slides where the
        smooth percentile stretch produces stromal texture instead of
        isolated nuclei.
        """
        from skimage.color import rgb2hed

        img = self.image
        if img.ndim != 3 or img.shape[2] < 3:
            raise ValueError("HematoxylinExtractor requires an RGB image")
        rgb = img[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        rgb_for_unmix = rgb
        if use_macenko:
            # Macenko-style normalization to a standard H&E reference.
            # Brings faded slides up to consistent stain intensity.
            try:
                normalized = preprocessing.normalize_he(rgb, Io=240, alpha=1, beta=0.15)
                normalized = _fix_he_swap(normalized, rgb)
                # Reproject normalized concentrations through the
                # canonical Ruifrok-Johnston H/E vectors so rgb2hed below
                # sees a canonical H&E image.
                ref_stain = np.array(
                    [[0.5626, 0.7201, 0.4062], [0.2159, 0.8012, 0.5581]]
                )
                recon_od = ref_stain.T @ normalized
                recon = np.clip(240.0 * np.exp(-recon_od), 0, 255).T.reshape(rgb.shape)
                rgb_for_unmix = recon.astype(np.uint8)
            except Exception:
                # Macenko can fail on degenerate (very faded / very dark)
                # tissue. Fall back to raw RGB rather than aborting.
                rgb_for_unmix = rgb

        hed = rgb2hed(rgb_for_unmix)
        h = hed[..., 0]  # hematoxylin channel (positive where stain is dense)

        if sparse:
            # Restrict the percentile to tissue pixels so a large dark
            # background doesn't pull the threshold down. Then zero
            # everything below the cutoff and stretch the survivors.
            from valis.preprocessing import create_tissue_mask_from_rgb

            try:
                _, tissue_mask = create_tissue_mask_from_rgb(rgb)
                tissue = tissue_mask > 0
            except Exception:
                tissue = np.ones(h.shape, dtype=bool)
            if tissue.sum() < 1000:
                tissue = np.ones(h.shape, dtype=bool)
            cutoff = np.percentile(h[tissue], sparse_pct)
            top = np.percentile(h[tissue], 99.9)
            denom = max(top - cutoff, 1e-6)
            out = np.clip((h - cutoff) / denom, 0.0, 1.0)
            out[~tissue] = 0
            if sparse_blur_sigma > 0:
                # Smooth so each surviving spot becomes a small Gaussian
                # blob with a real local maximum — gives keypoint matchers
                # something with scale to lock onto.
                from scipy.ndimage import gaussian_filter

                out = gaussian_filter(out.astype(np.float32), sparse_blur_sigma)
                m = out.max()
                if m > 0:
                    out = out / m
            return (out * 255).astype(np.uint8)

        lo, hi = np.percentile(h, (1, 99))
        if hi <= lo:
            hi = lo + 1e-6
        h = np.clip((h - lo) / (hi - lo), 0.0, 1.0)
        return (h * 255).astype(np.uint8)


def _fix_he_swap(normalized: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Detect and correct H/E row swaps from Macenko's angle heuristic.

    Hematoxylin stains pixels blue-purple, eosin pink-red. So the true
    hematoxylin concentration row should correlate positively with
    (B - R) across the image, while eosin should anti-correlate. If the
    rows are swapped, swap them back.
    """
    flat = rgb.reshape(-1, 3).astype(np.float32)
    blueness = flat[:, 2] - flat[:, 0]  # B - R
    # Mask to tissue pixels (anything sufficiently darker than 240 bg)
    od_proxy = 255.0 - flat.mean(axis=1)
    tissue = od_proxy > 15.0
    if tissue.sum() < 1000:
        return normalized
    b = blueness[tissue]
    if b.std() < 1e-3:
        return normalized
    c0 = np.corrcoef(normalized[0, tissue], b)[0, 1]
    c1 = np.corrcoef(normalized[1, tissue], b)[0, 1]
    if np.isnan(c0) or np.isnan(c1):
        return normalized
    if c1 > c0:
        return normalized[::-1].copy()
    return normalized


class Fluorescence(preprocessing.ImageProcesser):
    """Preprocess a single-band fluorescence image (e.g. DAPI) for matching.

    Nuclei are already bright on a dark background, so we keep polarity and
    apply only a simple percentile contrast stretch (default 1–99). We
    deliberately do NOT run adaptive histogram equalization (CLAHE) the way
    valis's ChannelGetter does: on the COMET/Xenium DAPI pair a plain
    percentile clip produced far more DISK+LightGlue matches than CLAHE
    (≈800 vs ≈520 filtered matches at 2048px). CLAHE amplifies local noise
    into spurious, non-repeatable keypoints that the matcher can't pair up
    across the two modalities.

    The tissue mask uses valis's multichannel path, which (unlike the RGB-only
    Luminosity mask) works on 1-band input.
    """

    def create_mask(self):
        from valis.preprocessing import create_tissue_mask_from_multichannel

        _, tissue_mask = create_tissue_mask_from_multichannel(self.image)
        return tissue_mask

    def process_image(self, *args, plo: float = 1.0, phi: float = 99.0, **kwargs):
        img = self.image
        if img.ndim == 3:
            img = img.mean(axis=-1)
        img = img.astype(np.float32)
        lo, hi = np.percentile(img, (plo, phi))
        if hi <= lo:
            hi = lo + 1e-6
        img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
        return (img * 255).astype(np.uint8)


class FluorescenceBlur(Fluorescence):
    """Tunable single-band fluorescence pipeline: :class:`Fluorescence`'s
    percentile stretch plus a set of optional classic CV steps.

    Steps run in this order; each is a no-op at its default except the
    Gaussian blur:

    1. ``bg_sigma`` — background subtraction: subtract a heavily
       Gaussian-blurred copy of the image to flatten uneven illumination /
       autofluorescence haze (0 = off).
    2. ``median`` — median filter kernel size, removes salt-and-pepper /
       hot pixels while keeping edges (0 or 1 = off).
    3. ``sigma`` — Gaussian blur, suppresses shot noise so the detector keys
       on nucleus-scale structure (0 = off).
    4. ``plo`` / ``phi`` — percentile contrast stretch to [0, 1].
    5. ``gamma`` — power-law curve; < 1 lifts dim nuclei, > 1 darkens them.
    6. ``clahe`` — adaptive histogram equalization with ``clahe_clip`` clip
       limit over a ``clahe_grid`` x ``clahe_grid`` tile grid. See the note
       on :class:`Fluorescence`: CLAHE can amplify noise into unmatched
       keypoints, so compare match counts with it on and off.
    7. ``unsharp_amount`` — unsharp mask with ``unsharp_radius`` to sharpen
       nucleus boundaries (0 = off).

    All radii / sigmas are in thumbnail pixels. With every optional step off
    and ``sigma=0`` this is identical to plain ``fluorescence``.
    """

    def process_image(
        self,
        *args,
        bg_sigma: float = 0.0,
        median: int = 0,
        sigma: float = 1.5,
        plo: float = 1.0,
        phi: float = 99.0,
        gamma: float = 1.0,
        clahe: bool = False,
        clahe_clip: float = 0.01,
        clahe_grid: int = 8,
        unsharp_amount: float = 0.0,
        unsharp_radius: float = 2.0,
        **kwargs,
    ):
        from scipy.ndimage import gaussian_filter, median_filter

        img = self.image
        if img.ndim == 3:
            img = img.mean(axis=-1)
        img = img.astype(np.float32)
        if bg_sigma > 0:
            img = np.maximum(img - gaussian_filter(img, bg_sigma), 0.0)
        median = int(median)
        if median > 1:
            img = median_filter(img, size=median)
        if sigma > 0:
            img = gaussian_filter(img, sigma)
        lo, hi = np.percentile(img, (plo, phi))
        if hi <= lo:
            hi = lo + 1e-6
        img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
        if gamma != 1.0:
            img = np.power(img, gamma)
        if clahe:
            from skimage import exposure

            grid = max(1, int(clahe_grid))
            kernel = (max(1, img.shape[0] // grid), max(1, img.shape[1] // grid))
            img = exposure.equalize_adapthist(
                img, kernel_size=kernel, clip_limit=clahe_clip
            ).astype(np.float32)
        if unsharp_amount > 0 and unsharp_radius > 0:
            blurred = gaussian_filter(img, unsharp_radius)
            img = np.clip(img + unsharp_amount * (img - blurred), 0.0, 1.0)
        return (img * 255).astype(np.uint8)


class InvertedFluorescence(preprocessing.ImageProcesser):
    """Reverse the inversion on an 'inverted DAPI' (or similar) greyscale image
    so that nuclei come out bright — matching the convention of hematoxylin
    deconvolution output.
    """

    def create_mask(self):
        img = self.image
        if img.ndim == 3:
            img = img.mean(axis=-1)
        img = img.astype(np.float32)
        # In an inverted-fluorescence image, tissue is dark on a bright bg.
        thresh = np.percentile(img, 90)
        mask = (img < thresh).astype(np.uint8) * 255
        return mask

    def process_image(self, *args, **kwargs):
        img = self.image
        if img.ndim == 3:
            img = img.mean(axis=-1)
        img = img.astype(np.float32)
        lo, hi = np.percentile(img, (1, 99))
        if hi <= lo:
            hi = lo + 1.0
        img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
        inverted = 1.0 - img
        return (inverted * 255).astype(np.uint8)


def pyvips_to_thumbnail_array(img: pyvips.Image, size: int) -> np.ndarray:
    """Render a pyvips image to a small greyscale numpy array.

    Uses pyvips' resize so we never materialize the full slide in memory.
    """
    # Never upsample: a larger "thumbnail" only costs memory downstream.
    scale = min(1.0, size / max(img.width, img.height))
    small = img.resize(scale)
    if small.bands > 1:
        small = small.colourspace("b-w")
    if small.format == "ushort":
        small = (small >> 8).cast("uchar")
    elif small.format != "uchar":
        small = small.cast("uchar")
    mem = small.write_to_memory()
    return np.frombuffer(mem, dtype=np.uint8).reshape(small.height, small.width)


def pyvips_to_thumbnail_rgb_array(img: pyvips.Image, size: int) -> np.ndarray:
    """Render a pyvips image to a small RGB numpy array. Used to feed
    color-aware preprocessors (HematoxylinExtractor, OD, ...) on a
    thumbnail.
    """
    # Never upsample: a larger "thumbnail" only costs memory downstream.
    scale = min(1.0, size / max(img.width, img.height))
    small = img.resize(scale)
    if small.format == "ushort":
        small = (small >> 8).cast("uchar")
    elif small.format != "uchar":
        small = small.cast("uchar")
    if small.bands == 1:
        small = small.bandjoin([small, small])
    elif small.bands > 3:
        small = small[0].bandjoin([small[1], small[2]])
    mem = small.write_to_memory()
    return np.frombuffer(mem, dtype=np.uint8).reshape(small.height, small.width, 3)


# ---------------------------------------------------------------------------
# Geometric pre-transform (flip + translate), applied before preprocessing
# ---------------------------------------------------------------------------

# Identity geometry. ``tx`` / ``ty`` are percentages of the image width /
# height (positive = content moves right / down) so the same setting means
# the same thing on a thumbnail and on the full-resolution slide. Flips are
# applied first, so the shift is in the flipped (displayed) frame. The canvas
# size is kept: content shifted off the edge is dropped, the exposed strip is
# filled with black.
GEOMETRY_DEFAULTS = {"flip_h": False, "flip_v": False, "tx": 0.0, "ty": 0.0}


def normalize_geometry(geometry) -> dict:
    g = {**GEOMETRY_DEFAULTS, **(geometry or {})}
    return {
        "flip_h": bool(g["flip_h"]),
        "flip_v": bool(g["flip_v"]),
        "tx": float(g["tx"]),
        "ty": float(g["ty"]),
    }


def geometry_is_identity(geometry) -> bool:
    return normalize_geometry(geometry) == normalize_geometry(None)


def _shift_px(pct: float, extent: int) -> int:
    return int(round(pct / 100.0 * extent))


def apply_geometry_array(arr: np.ndarray, geometry) -> np.ndarray:
    """Apply a flip/translate geometry to a 2-D or 3-D (H, W, C) array."""
    g = normalize_geometry(geometry)
    if g["flip_h"]:
        arr = arr[:, ::-1]
    if g["flip_v"]:
        arr = arr[::-1]
    h, w = arr.shape[:2]
    dx, dy = _shift_px(g["tx"], w), _shift_px(g["ty"], h)
    if dx or dy:
        out = np.zeros_like(arr)
        src_x0, dst_x0 = max(0, -dx), max(0, dx)
        src_y0, dst_y0 = max(0, -dy), max(0, dy)
        cw, ch = w - abs(dx), h - abs(dy)
        if cw > 0 and ch > 0:
            out[dst_y0 : dst_y0 + ch, dst_x0 : dst_x0 + cw] = arr[
                src_y0 : src_y0 + ch, src_x0 : src_x0 + cw
            ]
        arr = out
    return np.ascontiguousarray(arr)


def apply_geometry_pyvips(img: pyvips.Image, geometry) -> pyvips.Image:
    """Full-resolution counterpart of :func:`apply_geometry_array`."""
    g = normalize_geometry(geometry)
    if g["flip_h"]:
        img = img.fliphor()
    if g["flip_v"]:
        img = img.flipver()
    dx, dy = _shift_px(g["tx"], img.width), _shift_px(g["ty"], img.height)
    if abs(dx) >= img.width or abs(dy) >= img.height:
        # Shifted fully off-canvas (embed rejects this): all black.
        return (img * 0).cast(img.format)
    if dx or dy:
        img = img.embed(dx, dy, img.width, img.height, extend="black")
    return img


def geometry_tag(geometry) -> str:
    """Short filename-safe tag identifying a geometry (for cached copies)."""
    g = normalize_geometry(geometry)
    return (
        f"_fh{int(g['flip_h'])}_fv{int(g['flip_v'])}"
        f"_tx{g['tx']:+.1f}_ty{g['ty']:+.1f}"
    )


def run_processor_on_thumbnail(
    processor_spec, thumb_array: np.ndarray, src_f: str
) -> np.ndarray:
    """Instantiate a valis ``ImageProcesser`` subclass on a thumbnail and
    return its ``process_image`` output. ``processor_spec`` is a
    ``[cls, kwargs]`` pair as stored in :data:`PROCESSOR_REGISTRY`.
    """
    cls, kwargs = processor_spec
    proc = cls(image=thumb_array, src_f=src_f, level=0, series=0)
    return proc.process_image(**(kwargs or {}))


# ---------------------------------------------------------------------------
# Registry + declarative schemas
# ---------------------------------------------------------------------------

# Maps a user-facing stain/preprocessor name to a ``[ProcessorClass, kwargs]``
# pair. Base kwargs here are the "identity" defaults; the web app overrides
# them with per-slider params and the CLI uses the fixed variants.
PROCESSOR_REGISTRY = {
    "he-hematoxylin": [HematoxylinExtractor, {}],
    "he-hematoxylin-raw": [HematoxylinExtractor, {"use_macenko": False}],
    "he-hematoxylin-sparse": [HematoxylinExtractor, {"sparse": True}],
    "fluorescence": [Fluorescence, {}],
    "fluorescence-blur": [FluorescenceBlur, {}],
    "inverted-fluorescence": [InvertedFluorescence, {}],
    "od": [preprocessing.OD, {}],
    "colorful-standardizer": [preprocessing.ColorfulStandardizer, {}],
    "luminosity": [preprocessing.Luminosity, {}],
    "channel-getter": [preprocessing.ChannelGetter, {}],
}

# Whether each processor consumes an RGB thumbnail or a greyscale one. Color
# aware processors (HematoxylinExtractor.process_image raises on non-RGB, OD
# indexes axis 2) need RGB; single-band ones want greyscale.
PROCESSOR_INPUT = {
    "he-hematoxylin": "rgb",
    "he-hematoxylin-raw": "rgb",
    "he-hematoxylin-sparse": "rgb",
    "fluorescence": "gray",
    "fluorescence-blur": "gray",
    "inverted-fluorescence": "gray",
    "od": "rgb",
    "colorful-standardizer": "rgb",
    "luminosity": "rgb",
    "channel-getter": "gray",
}

# Declarative per-processor parameter schema. The web frontend renders these
# into a dropdown + dynamic sliders/toggles; the backend validates and forwards
# the values as ``process_image`` kwargs. Ranges are taken from the actual
# ``process_image`` signatures in this module and ``valis.preprocessing``.
PARAM_SCHEMA = {
    "he-hematoxylin": {
        "input": "rgb",
        "params": [
            {"name": "use_macenko", "type": "bool", "default": True},
            {"name": "sparse", "type": "bool", "default": False},
            {
                "name": "sparse_pct",
                "type": "float",
                "min": 80,
                "max": 97,
                "step": 1,
                "default": 90,
            },
            {
                "name": "sparse_blur_sigma",
                "type": "float",
                "min": 0,
                "max": 2.5,
                "step": 0.1,
                "default": 1.5,
            },
        ],
    },
    "he-hematoxylin-raw": {
        "input": "rgb",
        "params": [
            {"name": "sparse", "type": "bool", "default": False},
            {
                "name": "sparse_pct",
                "type": "float",
                "min": 80,
                "max": 97,
                "step": 1,
                "default": 90,
            },
            {
                "name": "sparse_blur_sigma",
                "type": "float",
                "min": 0,
                "max": 2.5,
                "step": 0.1,
                "default": 1.5,
            },
        ],
    },
    "he-hematoxylin-sparse": {
        "input": "rgb",
        "params": [
            {"name": "use_macenko", "type": "bool", "default": True},
            {
                "name": "sparse_pct",
                "type": "float",
                "min": 80,
                "max": 97,
                "step": 1,
                "default": 90,
            },
            {
                "name": "sparse_blur_sigma",
                "type": "float",
                "min": 0,
                "max": 2.5,
                "step": 0.1,
                "default": 1.5,
            },
        ],
    },
    "fluorescence": {
        "input": "gray",
        "params": [
            {
                "name": "plo",
                "type": "float",
                "min": 0,
                "max": 5,
                "step": 0.1,
                "default": 1,
            },
            {
                "name": "phi",
                "type": "float",
                "min": 95,
                "max": 100,
                "step": 0.1,
                "default": 99,
            },
        ],
    },
    "fluorescence-blur": {
        "input": "gray",
        "params": [
            {
                "name": "bg_sigma",
                "type": "float",
                "min": 0,
                "max": 100,
                "step": 1,
                "default": 0,
            },
            {
                "name": "median",
                "type": "int",
                "min": 0,
                "max": 9,
                "step": 1,
                "default": 0,
            },
            {
                "name": "sigma",
                "type": "float",
                "min": 0,
                "max": 5,
                "step": 0.1,
                "default": 1.5,
            },
            {
                "name": "plo",
                "type": "float",
                "min": 0,
                "max": 5,
                "step": 0.1,
                "default": 1,
            },
            {
                "name": "phi",
                "type": "float",
                "min": 95,
                "max": 100,
                "step": 0.1,
                "default": 99,
            },
            {
                "name": "gamma",
                "type": "float",
                "min": 0.3,
                "max": 3,
                "step": 0.05,
                "default": 1,
            },
            {"name": "clahe", "type": "bool", "default": False},
            {
                "name": "clahe_clip",
                "type": "float",
                "min": 0.001,
                "max": 0.05,
                "step": 0.001,
                "default": 0.01,
            },
            {
                "name": "clahe_grid",
                "type": "int",
                "min": 2,
                "max": 32,
                "step": 1,
                "default": 8,
            },
            {
                "name": "unsharp_amount",
                "type": "float",
                "min": 0,
                "max": 3,
                "step": 0.1,
                "default": 0,
            },
            {
                "name": "unsharp_radius",
                "type": "float",
                "min": 0.5,
                "max": 10,
                "step": 0.5,
                "default": 2,
            },
        ],
    },
    "inverted-fluorescence": {"input": "gray", "params": []},
    "od": {
        "input": "rgb",
        "params": [
            {"name": "adaptive_eq", "type": "bool", "default": False},
            {
                "name": "p",
                "type": "int",
                "min": 80,
                "max": 100,
                "step": 1,
                "default": 95,
            },
        ],
    },
    "colorful-standardizer": {
        "input": "rgb",
        "params": [
            {"name": "invert", "type": "bool", "default": True},
            {"name": "adaptive_eq", "type": "bool", "default": False},
        ],
    },
    "luminosity": {"input": "rgb", "params": []},
    "channel-getter": {
        "input": "gray",
        "params": [
            {"name": "adaptive_eq", "type": "bool", "default": False},
            {"name": "invert", "type": "bool", "default": False},
        ],
    },
}

# Global detector/matcher controls (not per image).
MATCHER_SCHEMA = {
    # "loma-b" uses its own DaD + DeDoDe-G features and ignores "detector".
    "matcher": {
        "type": "enum",
        "options": ["lightglue", "loma-b"],
        "default": "lightglue",
    },
    "detector": {"type": "enum", "options": ["disk", "dedode"], "default": "disk"},
    "max_keypoints": {
        "type": "int",
        "min": 512,
        "max": 7500,
        "step": 512,
        "default": 7500,
    },
    "ransac_thresh": {"type": "float", "min": 1, "max": 15, "step": 1, "default": 7},
    "filter_method": {
        "type": "enum",
        "options": ["magsac", "ransac", "none"],
        "default": "magsac",
    },
}

# Valid names for the ``--image-stain`` / ``--reference-stain`` CLI choices.
STAIN_CHOICES = (
    "auto",
    "he-hematoxylin",
    "he-hematoxylin-raw",
    "he-hematoxylin-sparse",
    "fluorescence",
    "fluorescence-blur",
    "inverted-fluorescence",
    "od",
    "colorful-standardizer",
    "luminosity",
)


def resolve_auto_stain(path: str) -> str:
    """Pick a real processor name for ``auto`` based on band count + mean
    intensity so 'nuclei = bright' on both sides, which is what the matcher
    needs. RGB -> hematoxylin; bright single-band -> un-invert; dark
    single-band -> plain fluorescence stretch.
    """
    peek = pyvips.Image.new_from_file(path, page=0)
    thumb = pyvips_to_thumbnail_array(peek, 256)
    mean = float(thumb.mean())
    if peek.bands >= 3:
        return "he-hematoxylin"
    return "inverted-fluorescence" if mean > 127 else "fluorescence"


# Moving-image geometric pre-transform controls. Only the moving image gets
# these: the reference defines the output frame of the aligned result.
GEOMETRY_SCHEMA = [
    {"name": "flip_h", "type": "bool", "default": False},
    {"name": "flip_v", "type": "bool", "default": False},
    {"name": "tx", "type": "float", "min": -50, "max": 50, "step": 0.5, "default": 0},
    {"name": "ty", "type": "float", "min": -50, "max": 50, "step": 0.5, "default": 0},
]


# Working resolution: longest side, in px, of the thumbnail each image is
# preprocessed and matched at (never above native). Set per image, since two
# modalities can have very different pixel sizes. Alignment uses the matches
# found at exactly this resolution.
RESOLUTION_SCHEMA = {
    "type": "int",
    "min": 256,
    "max": 4096,
    "step": 128,
    "default": 1024,
}

# Remaining knobs of the full alignment run.
ALIGNMENT_SCHEMA = {
    # Minimum matches for the pre-aligned check (what valis's rigid step
    # sees) and for valis's own rigid matching.
    "min_matches": {"type": "int", "min": 3, "max": 200, "step": 1, "default": 30},
    # Longest side, in px, of the images valis registers (rigid + non-rigid).
    # The pre-aligned check in the preview matches at this resolution too.
    "valis_resolution": {
        "type": "int",
        "min": 512,
        "max": 4096,
        "step": 128,
        "default": 1024,
    },
}


def public_schema() -> dict:
    """Return the JSON-serializable schema for the frontend (no class refs)."""
    return {
        "processors": PARAM_SCHEMA,
        "matcher": MATCHER_SCHEMA,
        "geometry": GEOMETRY_SCHEMA,
        "resolution": RESOLUTION_SCHEMA,
        "alignment": ALIGNMENT_SCHEMA,
    }
