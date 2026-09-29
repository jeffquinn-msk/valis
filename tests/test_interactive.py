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
    assert set(schema) == {"processors", "matcher", "geometry"}


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
        {"tx": 12.5, "ty": -20},
        {"flip_h": True, "flip_v": True, "tx": -30, "ty": 7.5},
        {"tx": 100},
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
    got = np.ndarray(
        buffer=got.write_to_memory(), dtype=np.uint8, shape=(37, 53, 3)
    )
    np.testing.assert_array_equal(got, expected)


def test_geometry_semantics():
    a = np.arange(12, dtype=np.uint8).reshape(3, 4)
    np.testing.assert_array_equal(
        processors.apply_geometry_array(a, {"flip_h": True}), a[:, ::-1]
    )
    # +25% of width 4 = 1 px right; exposed column is black
    shifted = processors.apply_geometry_array(a, {"tx": 25})
    np.testing.assert_array_equal(shifted[:, 1:], a[:, :-1])
    assert (shifted[:, 0] == 0).all()
    assert processors.geometry_is_identity({})
    assert not processors.geometry_is_identity({"ty": 1})
