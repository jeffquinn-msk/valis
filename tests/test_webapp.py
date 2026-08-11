"""Tests for the interactive alignment web app backend."""

import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from valis.webapp import app as webapp_app  # noqa: E402
from valis.webapp import browse  # noqa: E402


@pytest.fixture()
def data_root(tmp_path):
    import pyvips

    root = str(tmp_path)
    rng = np.random.default_rng(0)

    rgb = rng.integers(120, 220, size=(240, 200, 3)).astype(np.uint8)
    pyvips.Image.new_from_memory(rgb.tobytes(), 200, 240, 3, "uchar").tiffsave(
        os.path.join(root, "moving.ome.tif"), tile=True, pyramid=True
    )

    g = np.zeros((220, 210), np.uint8)
    for _ in range(40):
        y, x = rng.integers(5, 215), rng.integers(5, 205)
        g[y - 3 : y + 3, x - 3 : x + 3] = 255
    pyvips.Image.new_from_memory(g.tobytes(), 210, 220, 1, "uchar").tiffsave(
        os.path.join(root, "reference.ome.tif"), tile=True, pyramid=True
    )

    # subdirectory to exercise the browser
    os.makedirs(os.path.join(root, "sub"), exist_ok=True)
    return root


@pytest.fixture()
def client(data_root, monkeypatch):
    monkeypatch.setattr(webapp_app, "DATA_ROOT", data_root)
    return TestClient(webapp_app.app)


def test_import_order_no_segfault():
    """Importing the webapp app then torch must not segfault (exit 139)."""
    code = "from valis.webapp import app; import torch; print('ok')"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert (
        proc.returncode == 0
    ), f"import order regressed: rc={proc.returncode}\n{proc.stderr.decode()[-500:]}"


def test_browse_is_sandboxed():
    with pytest.raises(browse.BrowseError):
        browse.resolve_within_root("/some/root", "../../etc/passwd")


def test_schema_endpoint(client):
    schema = client.get("/api/schema").json()
    assert "processors" in schema and "matcher" in schema
    assert "od" in schema["processors"]


def test_browse_endpoint(client):
    data = client.get("/api/browse").json()
    names = {f["name"] for f in data["files"]}
    assert {"moving.ome.tif", "reference.ome.tif"} <= names
    assert any(d["name"] == "sub" for d in data["dirs"])
    # escape attempt rejected
    assert client.get("/api/browse", params={"path": "../.."}).status_code == 400


def test_session_thumbnail_preprocess(client):
    sess = client.post(
        "/api/session",
        json={"image_path": "moving.ome.tif", "reference_path": "reference.ome.tif"},
    ).json()
    assert sess["image"]["bands"] == 3
    assert sess["reference"]["suggested_stain"] == "fluorescence"
    sid = sess["session_id"]

    thumb = client.get(f"/api/thumbnail/{sid}/image", params={"size": 128})
    assert thumb.status_code == 200
    assert thumb.headers["content-type"] == "image/png"

    pre = client.post(
        f"/api/preprocess/{sid}/reference",
        json={
            "processor": "fluorescence",
            "params": {"plo": 1, "phi": 99},
            "size": 128,
        },
    )
    assert pre.status_code == 200 and pre.headers["content-type"] == "image/png"


def test_match_endpoint_runs(client):
    """The fast path must run without the kp2_xy=None crash (may find 0 matches
    on synthetic data — we only assert it returns a well-formed payload)."""
    sess = client.post(
        "/api/session",
        json={"image_path": "moving.ome.tif", "reference_path": "reference.ome.tif"},
    ).json()
    sid = sess["session_id"]
    res = client.post(
        f"/api/match/{sid}",
        json={
            "image": {"processor": "od", "params": {}},
            "reference": {"processor": "fluorescence", "params": {}},
            "matcher": {
                "detector": "disk",
                "max_keypoints": 2048,
                "ransac_thresh": 7,
                "filter_method": "magsac",
            },
            "size": 128,
        },
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert set(data) >= {"matches", "n_total", "n_filtered", "image_size"}
    assert len(data["matches"]["kp1"]) == len(data["matches"]["kp2"])


def test_detect_and_match_pairs_are_equal_length():
    """Directly exercise the fast path: matching a textured image against
    itself must find matches and return equal-length paired arrays (regression
    for match_images crashing when kp2_xy is not pre-detected)."""
    from valis.webapp import matching

    rng = np.random.default_rng(7)
    # Structured texture with repeatable corners for DISK to lock onto.
    img = rng.integers(0, 255, size=(256, 256), dtype=np.uint8)
    img = np.clip(img.astype(np.int32) + 20, 0, 255).astype(np.uint8)

    kp1, kp2, n_total, n_filtered = matching.detect_and_match(
        img,
        img.copy(),
        detector="disk",
        max_keypoints=2048,
        ransac_thresh=7,
        filter_method="magsac",
    )
    assert kp1.shape == kp2.shape
    assert kp1.shape[1] == 2
    assert n_total >= n_filtered
    assert n_filtered > 0, "identical images should yield matches"


def test_result_supports_range_requests(client, data_root, monkeypatch):
    """/api/result must serve HTTP Range (GeoTIFFTileSource requires it)."""
    tif = os.path.join(data_root, "moving.ome.tif")
    monkeypatch.setitem(
        webapp_app._jobs,
        "fakejob",
        {
            "state": "done",
            "stage": "done",
            "progress": 1.0,
            "message": "",
            "result": tif,
        },
    )
    r = client.get(
        "/api/result/fakejob/aligned.ome.tif", headers={"Range": "bytes=0-99"}
    )
    assert r.status_code == 206
    assert "content-range" in {k.lower() for k in r.headers}
    assert len(r.content) == 100
