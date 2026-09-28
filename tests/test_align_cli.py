"""Tests for the ``scripts/align_two_images.py`` CLI argument handling."""

import importlib.util
import os

import pytest

SCRIPT = os.path.join(
    os.path.dirname(__file__), os.pardir, "scripts", "align_two_images.py"
)


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("align_two_images", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse(cli, *extra):
    return cli.get_parser().parse_args(
        ["--reference", "r.tif", "--image", "i.tif", "--output-dir", "out", *extra]
    )


def test_no_matcher_flags_keeps_valis_default(cli):
    assert cli.matcher_cfg_from_args(_parse(cli)) is None


def test_matcher_flags_fill_unset_from_schema_defaults(cli):
    cfg = cli.matcher_cfg_from_args(
        _parse(cli, "--detector", "dedode", "--ransac-thresh", "5")
    )
    assert cfg == {
        "matcher": "lightglue",
        "detector": "dedode",
        "max_keypoints": 7500,
        "ransac_thresh": 5.0,
        "filter_method": "magsac",
    }


def test_matcher_flag_selects_loma(cli):
    cfg = cli.matcher_cfg_from_args(_parse(cli, "--matcher", "loma-b"))
    assert cfg["matcher"] == "loma-b"


def test_filter_method_rejects_preview_only_none(cli):
    with pytest.raises(SystemExit):
        _parse(cli, "--filter-method", "none")
