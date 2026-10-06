"""Tests for the shared ``valis.interactive`` package (processors + schema).

conftest.py imports valis first, so importing pyvips/torch-backed modules here
is safe (valis-before-torch ordering).
"""

import numpy as np
import pytest

from valis.interactive import processors


def _rgb(h=96, w=80):
    rng = np.random.default_rng(1)
    base = rng.integers(150, 230, size=(h, w, 3)).astype(np.uint8)
    # add a few darker "nuclei" blobs so color-aware processors have signal
    for _ in range(15):
        y, x = rng.integers(5, h - 5), rng.integers(5, w - 5)
        base[y - 3 : y + 3, x - 3 : x + 3] = [90, 70, 160]
    return base


def _gray(h=96, w=80):
    rng = np.random.default_rng(2)
    g = np.zeros((h, w), np.uint8)
    for _ in range(20):
        y, x = rng.integers(3, h - 3), rng.integers(3, w - 3)
        g[y - 2 : y + 2, x - 2 : x + 2] = 255
    return g


@pytest.fixture(scope="module")
def sample_files(tmp_path_factory):
    """Real on-disk tifs — the ImageProcesser base reads ``src_f`` as a slide."""
    import pyvips

    d = tmp_path_factory.mktemp("proc_inputs")
    rgb, gray = _rgb(), _gray()
    rgb_f = str(d / "rgb.tif")
    gray_f = str(d / "gray.tif")
    pyvips.Image.new_from_memory(
        rgb.tobytes(), rgb.shape[1], rgb.shape[0], 3, "uchar"
    ).tiffsave(rgb_f)
    pyvips.Image.new_from_memory(
        gray.tobytes(), gray.shape[1], gray.shape[0], 1, "uchar"
    ).tiffsave(gray_f)
    return {"rgb": (rgb, rgb_f), "gray": (gray, gray_f)}


@pytest.mark.parametrize("name", list(processors.PARAM_SCHEMA.keys()))
def test_processor_defaults_produce_uint8_2d(name, sample_files):
    spec = processors.PARAM_SCHEMA[name]
    thumb, src_f = sample_files[spec["input"]]
    defaults = {p["name"]: p["default"] for p in spec["params"]}
    cls, base_kw = processors.PROCESSOR_REGISTRY[name]
    out = processors.run_processor_on_thumbnail(
        [cls, {**base_kw, **defaults}], thumb, src_f
    )
    assert out.ndim == 2, f"{name} should return a single-channel image"
    assert out.dtype == np.uint8, f"{name} should return uint8"
    assert out.shape == thumb.shape[:2]


def test_registry_and_schema_keys_match():
    # Every schema entry must have a registry class and an input declaration.
    for name in processors.PARAM_SCHEMA:
        assert name in processors.PROCESSOR_REGISTRY
        assert name in processors.PROCESSOR_INPUT


def test_public_schema_is_json_serializable():
    import json

    schema = processors.public_schema()
    json.dumps(schema)  # must not raise (no class refs leak through)
    assert set(schema) == {
        "processors", "matcher", "geometry", "resolution", "alignment"
    }


def test_thumbnail_helpers_downsample():
    import pyvips

    arr = _rgb(400, 300)
    vi = pyvips.Image.new_from_memory(arr.tobytes(), 300, 400, 3, "uchar")
    gray = processors.pyvips_to_thumbnail_array(vi, 128)
    rgb = processors.pyvips_to_thumbnail_rgb_array(vi, 128)
    assert gray.ndim == 2 and max(gray.shape) == 128
    assert rgb.ndim == 3 and rgb.shape[2] == 3 and max(rgb.shape[:2]) == 128


@pytest.mark.parametrize(
    "geometry",
    [
        {"flip_h": True},
        {"flip_v": True},
        {"rotate": 90},
        {"rotate": 180},
        {"rotate": "270"},
        {"flip_h": True, "rotate": 90},
        {"flip_h": True, "flip_v": True, "rotate": 270},
    ],
)
def test_geometry_array_matches_pyvips(geometry):
    """The thumbnail (numpy) and full-res (pyvips) pre-transforms must agree,
    or the preview would show a different image than the one registered."""
    import pyvips

    arr = _rgb(37, 53)
    vi = pyvips.Image.new_from_memory(arr.tobytes(), 53, 37, 3, "uchar")
    expected = processors.apply_geometry_array(arr, geometry)
    got = processors.apply_geometry_pyvips(vi, geometry)
    assert (got.width, got.height) == processors.geometry_output_wh(geometry, (53, 37))
    got = np.ndarray(
        buffer=got.write_to_memory(), dtype=np.uint8, shape=expected.shape
    )
    np.testing.assert_array_equal(got, expected)


def test_geometry_semantics():
    a = np.arange(12, dtype=np.uint8).reshape(3, 4)
    np.testing.assert_array_equal(
        processors.apply_geometry_array(a, {"flip_h": True}), a[:, ::-1]
    )
    # clockwise: the top-left corner ends up top-right
    rotated = processors.apply_geometry_array(a, {"rotate": 90})
    assert rotated.shape == (4, 3)
    assert rotated[0, -1] == a[0, 0]
    np.testing.assert_array_equal(rotated, np.rot90(a, k=-1))
    assert processors.geometry_is_identity({})
    assert processors.geometry_is_identity({"rotate": 360})
    assert not processors.geometry_is_identity({"rotate": 90})
    with pytest.raises(ValueError):
        processors.normalize_geometry({"rotate": 45})


def test_fluorescence_blur_invert_matches_uninverted_original(tmp_path):
    """invert on an inverted image must give what plain processing gives on
    the original, and the tissue mask valis builds afterwards must follow."""
    rng = np.random.default_rng(0)
    img = np.zeros((64, 64), np.uint8)
    img[16:48, 16:48] = rng.integers(80, 255, size=(32, 32))  # bright tissue
    inverted = 255 - img

    import pyvips

    src_f = str(tmp_path / "src.tif")  # the processor opens its source file
    pyvips.Image.new_from_array(img).write_to_file(src_f)

    def run(arr, **params):
        proc = processors.FluorescenceBlur(arr, src_f=src_f, level=0, series=0)
        return proc.process_image(**params), proc.create_mask()

    plain, plain_mask = run(img, sigma=1)
    flipped, flipped_mask = run(inverted, sigma=1, invert=True)
    np.testing.assert_allclose(flipped.astype(int), plain.astype(int), atol=1)
    np.testing.assert_array_equal(flipped_mask, plain_mask)
    assert {"name": "invert", "type": "bool", "default": False} in (
        processors.PARAM_SCHEMA["fluorescence-blur"]["params"]
    )
