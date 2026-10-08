"""Tests for the ``valis-align`` / ``valis-match`` CLI argument handling."""

import pytest

from valis.cli import align, common, match


def _parse(*extra):
    return align.get_parser().parse_args(
        ["--reference", "r.tif", "--image", "i.tif", "--output-dir", "out", *extra]
    )


def test_no_matcher_flags_keeps_valis_default():
    assert common.matcher_cfg_from_args(_parse()) is None


def test_matcher_flags_fill_unset_from_schema_defaults():
    cfg = common.matcher_cfg_from_args(
        _parse("--detector", "dedode", "--ransac-thresh", "5")
    )
    assert cfg == {
        "matcher": "lightglue",
        "detector": "dedode",
        "max_keypoints": 7500,
        "ransac_thresh": 5.0,
        "filter_method": "magsac",
    }


def test_always_fills_matcher_defaults():
    cfg = common.matcher_cfg_from_args(_parse(), always=True)
    assert cfg["matcher"] == "lightglue" and cfg["detector"] == "disk"


def test_matcher_flag_selects_loma():
    cfg = common.matcher_cfg_from_args(_parse("--matcher", "loma-b"))
    assert cfg["matcher"] == "loma-b"


def test_matcher_flag_selects_romav2():
    cfg = common.matcher_cfg_from_args(_parse("--matcher", "romav2"))
    assert cfg["matcher"] == "romav2"


def test_filter_method_accepts_none():
    cfg = common.matcher_cfg_from_args(_parse("--filter-method", "none"))
    assert cfg["filter_method"] == "none"


def test_stain_flags_still_work():
    args = _parse("--image-stain", "he-hematoxylin", "--reference-stain", "od")
    assert args.image_processor == "he-hematoxylin"
    assert args.reference_processor == "od"


def test_params_are_typed_from_schema():
    params = common.parse_params(
        "fluorescence-blur", ["invert=true", "median=3", "sigma=0.5"]
    )
    assert params == {"invert": True, "median": 3, "sigma": 0.5}


def test_enum_param():
    assert common.parse_params("color-range", ["intensity=mask"]) == {
        "intensity": "mask"
    }
    with pytest.raises(ValueError, match="one of"):
        common.parse_params("color-range", ["intensity=bogus"])


@pytest.mark.parametrize(
    "processor,item,msg",
    [
        ("od", "nope=1", "no parameter"),
        ("od", "p", "NAME=VALUE"),
        ("od", "p=abc", "expected int"),
        ("od", "adaptive_eq=maybe", "expected bool"),
        ("auto", "p=95", "explicitly"),
    ],
)
def test_bad_params_rejected(processor, item, msg):
    with pytest.raises(ValueError, match=msg):
        common.parse_params(processor, [item])


def test_side_settings_reports_bad_param_via_parser():
    parser = align.get_parser()
    args = parser.parse_args(
        ["--image", "i", "--reference", "r", "--output-dir", "o"]
        + ["--image-stain", "od", "--image-param", "nope=1"]
    )
    with pytest.raises(SystemExit):
        common.side_settings(parser, args)


def test_geometry_and_sizes():
    args = match.get_parser().parse_args(
        ["--flip-h", "--rotate", "270", "--size", "512", "--reference-size", "800"]
    )
    assert common.geometry_from_args(args) == {
        "flip_h": True,
        "flip_v": False,
        "rotate": 270,
    }
    assert common.size_for(args, "image") == 512
    assert common.size_for(args, "reference") == 800


def test_rotate_must_be_multiple_of_90():
    with pytest.raises(SystemExit):
        match.get_parser().parse_args(["--rotate", "45"])


def test_list_processors(capsys):
    match.main(["--list-processors"])
    out = capsys.readouterr().out
    assert "fluorescence-blur" in out and "sparse_pct" in out
