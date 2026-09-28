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


def public_schema() -> dict:
    """Return the JSON-serializable schema for the frontend (no class refs)."""
    return {"processors": PARAM_SCHEMA, "matcher": MATCHER_SCHEMA}
