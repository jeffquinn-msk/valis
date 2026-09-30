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

    blurred = client.post(
        f"/api/preprocess/{sid}/reference",
        json={
            "processor": "fluorescence-blur",
            "params": {"sigma": 2.5, "plo": 1, "phi": 99},
            "size": 128,
        },
    )
    assert blurred.status_code == 200
    assert blurred.headers["content-type"] == "image/png"
    assert blurred.content != pre.content

    everything = client.post(
        f"/api/preprocess/{sid}/reference",
        json={
            "processor": "fluorescence-blur",
            "params": {
                "bg_sigma": 20,
                "median": 3,
                "sigma": 1,
                "gamma": 0.7,
                "clahe": True,
                "clahe_clip": 0.02,
                "clahe_grid": 4,
                "unsharp_amount": 1,
                "unsharp_radius": 2,
            },
            "size": 128,
        },
    )
    assert everything.status_code == 200
    assert everything.headers["content-type"] == "image/png"


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


def _fake_detect_and_match(n=40):
    def fake(img1, img2, **kwargs):
        rng = np.random.default_rng(1)
        kp1 = rng.uniform(0, min(img1.shape[:2]) - 1, size=(n, 2))
        return kp1, kp1 * 1.0, n + 10, n

    return fake


def _wait_for_job(client, job_id):
    import time

    for _ in range(100):
        status = client.get(f"/api/align/{job_id}/status").json()
        if status["state"] in ("done", "error"):
            return status
        time.sleep(0.05)
    return status


def _new_session(client):
    return client.post(
        "/api/session",
        json={"image_path": "moving.ome.tif", "reference_path": "reference.ome.tif"},
    ).json()["session_id"]


def test_align_uses_exactly_the_displayed_matches(client, monkeypatch):
    """Alignment must run on the preview's own matches, preprocessing and
    per-image resolution -- nothing recomputed differently under the hood."""
    captured = {}

    def fake_run_alignment(**kwargs):
        captured.update(kwargs)
        return "/nonexistent/aligned.ome.tif"

    monkeypatch.setattr(webapp_app.pipeline, "run_alignment", fake_run_alignment)
    monkeypatch.setattr(
        webapp_app.matching, "detect_and_match", _fake_detect_and_match()
    )
    sid = _new_session(client)
    matcher = {
        "matcher": "loma-b",
        "detector": "dedode",
        "max_keypoints": 2048,
        "ransac_thresh": 5,
        "filter_method": "ransac",
    }
    geometry = {"flip_h": False, "flip_v": True, "rotate": "90"}
    match = client.post(
        f"/api/match/{sid}",
        json={
            "image": {
                "processor": "od", "params": {}, "geometry": geometry, "size": 128,
            },
            "reference": {"processor": "fluorescence", "params": {}, "size": 96},
            "matcher": matcher,
        },
    ).json()
    assert max(match["image_size"]) == 128
    assert max(match["reference_size"]) == 96

    job_id = client.post(
        f"/api/align/{sid}",
        json={
            "match_id": match["match_id"],
            "min_matches": 10,
            "valis_resolution": 1024,
        },
    ).json()["job_id"]
    status = _wait_for_job(client, job_id)
    assert status["state"] == "done", status

    preview = captured["preview_match"]
    np.testing.assert_allclose(preview.kp_moving, match["matches"]["kp1"])
    np.testing.assert_allclose(preview.kp_reference, match["matches"]["kp2"])
    assert max(preview.moving_img.shape[:2]) == 128
    assert max(preview.reference_img.shape[:2]) == 96
    assert captured["matcher_cfg"] == matcher
    assert captured["image_geometry"] == geometry
    assert captured["image_stain"] == "od"
    assert captured["max_processed_image_dim_px"] == 1024


def test_align_requires_a_displayed_match(client):
    sid = _new_session(client)
    res = client.post(f"/api/align/{sid}", json={"match_id": "nope"})
    assert res.status_code == 400


def test_align_refuses_below_min_matches(client, monkeypatch):
    called = []
    monkeypatch.setattr(
        webapp_app.pipeline, "run_alignment", lambda **kw: called.append(kw)
    )
    monkeypatch.setattr(
        webapp_app.matching, "detect_and_match", _fake_detect_and_match(n=5)
    )
    sid = _new_session(client)
    match = client.post(
        f"/api/match/{sid}",
        json={
            "image": {"processor": "od", "params": {}},
            "reference": {"processor": "fluorescence", "params": {}},
            "size": 128,
        },
    ).json()
    job_id = client.post(
        f"/api/align/{sid}", json={"match_id": match["match_id"], "min_matches": 30}
    ).json()["job_id"]
    status = _wait_for_job(client, job_id)
    assert status["state"] == "error"
    assert "5 matches" in status["message"]
    assert not called


def test_preview_affine_moves_pixels_to_their_matches():
    """The preview fit, lifted to full res and applied with vips, must land
    a moving-image feature on its reference position."""
    import pyvips
    from valis.interactive import pipeline

    # Thumbnails at 1/4 scale; reference = moving rotated 10 deg + shifted.
    T = pipeline.transform.SimilarityTransform(
        rotation=np.deg2rad(10), translation=(6, -3)
    )
    mov_xy = np.random.default_rng(3).uniform(10, 40, size=(30, 2))
    preview = pipeline.PreviewMatch(
        kp_moving=mov_xy,
        kp_reference=T(mov_xy),
        moving_img=np.zeros((50, 60), np.uint8),
        reference_img=np.zeros((50, 60), np.uint8),
    )
    A, T_fit = pipeline.preview_affine_full_res(preview, (240, 200), (240, 200))
    np.testing.assert_allclose(T_fit, T.params, atol=1e-6)

    img = np.zeros((200, 240), np.uint8)
    img[99:102, 79:82] = 255  # 3x3 block centred on (80, 100)
    warped = pipeline.apply_affine_pyvips(
        pyvips.Image.new_from_memory(img.tobytes(), 240, 200, 1, "uchar"),
        A,
        (240, 200),
    ).numpy()
    assert warped.shape == (200, 240)
    ys, xs = np.nonzero(warped > 64)
    expected = pipeline.transform.AffineTransform(matrix=A)([[80, 100]])[0]
    np.testing.assert_allclose([xs.mean(), ys.mean()], expected, atol=0.75)


def test_preview_affine_uses_rotated_moving_size():
    """A 90 deg rotation swaps the moving image's width and height; the
    preview was matched on the rotated thumbnail, so the lift to full res
    must use the rotated size. Here reference = rotated moving, so A = I."""
    from valis.interactive import pipeline, processors

    geometry = {"rotate": 90}
    mov_wh = (240, 200)  # before rotation
    rot_wh = processors.geometry_output_wh(geometry, mov_wh)
    assert rot_wh == (200, 240)
    xy = np.random.default_rng(4).uniform(5, 45, size=(30, 2))
    thumb = np.zeros((rot_wh[1] // 4, rot_wh[0] // 4), np.uint8)
    preview = pipeline.PreviewMatch(
        kp_moving=xy, kp_reference=xy.copy(), moving_img=thumb, reference_img=thumb
    )
    A, _ = pipeline.preview_affine_full_res(preview, rot_wh, rot_wh)
    np.testing.assert_allclose(A, np.eye(3), atol=1e-6)


def test_build_matcher_uses_requested_settings():
    from valis import feature_detectors, feature_matcher
    from valis.interactive import pipeline

    mat = pipeline.build_matcher(
        detector="disk", max_keypoints=1024, ransac_thresh=5, filter_method="ransac"
    )
    assert isinstance(mat, feature_matcher.LightGlueMatcher)
    assert isinstance(mat.feature_detector, feature_detectors.DiskFD)
    assert mat.match_filter_method == feature_matcher.RANSAC_NAME
    assert mat.ransac_thresh == 5
    # "none" is preview-only; the matcher itself still filters with MAGSAC.
    none_mat = pipeline.build_matcher(filter_method="none")
    assert none_mat.match_filter_method == feature_matcher.USAC_MAGSAC_NAME


def test_loma_matcher_rejects_other_features():
    """LoMa was trained on DaD + DeDoDe-G descriptors; pairing it with any
    other detector must fail loudly rather than produce garbage matches."""
    from valis import feature_detectors, feature_matcher
    from valis.interactive import pipeline

    with pytest.raises(TypeError, match="LoMaFD"):
        feature_matcher.LoMaMatcher(feature_detectors.DiskFD(num_features=512))
    with pytest.raises(ValueError, match="unknown matcher"):
        pipeline.build_detector(matcher="superglue")


def test_romav2_matcher_rejects_other_features():
    """RoMa v2's model lives in RoMaV2FD; any other detector can't match."""
    from valis import feature_detectors, feature_matcher

    with pytest.raises(TypeError, match="RoMaV2FD"):
        feature_matcher.RoMaV2Matcher(feature_detectors.DiskFD(num_features=512))


def _fake_romav2_fd(sampled):
    """A RoMaV2FD whose model returns ``sampled`` (N, 4) normalized matches,
    without loading any weights."""
    import torch
    from valis import feature_detectors

    class FakeModel:
        def match(self, img_a, img_b):
            assert img_a.shape[1] == 3 and img_b.shape[1] == 3  # RGB tensors
            return {}

        def sample(self, preds, num):
            m = torch.as_tensor(sampled, dtype=torch.float32)[:num]
            return m, torch.full((len(m),), 0.9), None, None

    fd = object.__new__(feature_detectors.RoMaV2FD)
    feature_detectors.FeatureDD.__init__(fd)
    fd.device, fd.model, fd.num_features = torch.device("cpu"), FakeModel(), 100
    return fd


def test_romav2_model_is_loaded_once_across_match_counts(monkeypatch):
    """Changing max keypoints only changes RoMa v2's sample count; it must
    not load another copy of the model."""
    from valis.webapp import matching

    built = []

    def fake_build(detector, max_keypoints, matcher):
        built.append(max_keypoints)
        return _fake_romav2_fd([])

    monkeypatch.setattr(matching, "_detectors", {})
    monkeypatch.setattr(matching.pipeline, "build_detector", fake_build)
    det_a, _ = matching._get_detector("disk", 1024, "romav2")
    det_b, _ = matching._get_detector("dedode", 4096, "romav2")
    assert det_a is det_b and built == [1024]
    assert det_b.num_features == 4096


def test_romav2_fd_returns_pixel_centre_coordinates():
    # normalized -1 / +1 are pixel edges: the first/last pixel centres of a
    # 100 px wide image are at -1 + 1/100 and 1 - 1/100.
    fd = _fake_romav2_fd([[-0.99, -0.98, 0.99, 0.0]])
    kp1, kp2, cert = fd.match(np.zeros((50, 100), np.uint8), np.zeros((20, 100), np.uint8))
    np.testing.assert_allclose(kp1, [[0.0, 0.0]], atol=1e-4)
    np.testing.assert_allclose(kp2, [[99.0, 9.5]], atol=1e-4)
    np.testing.assert_allclose(cert, [0.9])
    kp, desc = fd.detect_and_compute(np.zeros((50, 100), np.uint8))
    assert kp.shape == (0, 2) and desc.shape == (0, 1)


def test_romav2_matcher_matches_images_not_keypoints():
    """Keypoints passed in (valis passes RoMaV2FD's empty ones) are ignored;
    the dense matches come back as MatchInfo and go through RANSAC."""
    from valis import feature_matcher

    rng = np.random.default_rng(0)
    xy1 = rng.uniform(-0.9, 0.9, size=(60, 2))
    xy2 = xy1 * 0.5 + 0.1  # a similarity transform
    xy2[:5] = rng.uniform(-0.9, 0.9, size=(5, 2))  # outliers
    mat = feature_matcher.RoMaV2Matcher(
        _fake_romav2_fd(np.hstack([xy1, xy2])),
        match_filter_method=feature_matcher.RANSAC_NAME,
        ransac_thresh=2,
    )
    img = np.zeros((200, 200), np.uint8)
    empty_kp, empty_desc = np.empty((0, 2)), np.empty((0, 1))
    raw12, filt12, raw21, filt21 = mat.match_images(
        img, img, desc1=empty_desc, kp1_xy=empty_kp, desc2=empty_desc, kp2_xy=empty_kp
    )
    assert raw12.n_matches == 60
    assert 50 <= filt12.n_matches <= 55
    np.testing.assert_allclose(raw12.match_distances, 0.1, atol=1e-6)
    np.testing.assert_array_equal(filt21.matched_kp1_xy, filt12.matched_kp2_xy)


def test_geotiff_plugin_worker_is_served(client):
    """geotiff-tilesource fetches its decoding worker from the absolute path
    /assets/...; without it the result viewer opens but never draws tiles."""
    import glob

    workers = glob.glob(
        os.path.join(webapp_app.STATIC_DIR, "vendor", "assets", "tiff.worker-*.js")
    )
    assert workers, "vendored geotiff-tilesource worker is missing"
    r = client.get(f"/assets/{os.path.basename(workers[0])}")
    assert r.status_code == 200
    assert "javascript" in r.headers["content-type"]


def test_lightglue_filter_uses_ransac_thresh(monkeypatch):
    """LightGlueMatcher must pass its ransac_thresh to the outlier filter
    (it used to silently use the default of 7)."""
    from valis import feature_matcher
    from valis.webapp import matching

    seen = []
    real = feature_matcher.filter_matches_ransac

    def spy(*args, **kwargs):
        seen.append(kwargs.get("ransac_val"))
        return real(*args, **kwargs)

    monkeypatch.setattr(feature_matcher, "filter_matches_ransac", spy)
    rng = np.random.default_rng(7)
    img = rng.integers(0, 255, size=(256, 256), dtype=np.uint8)
    matching.detect_and_match(
        img, img.copy(), max_keypoints=512, ransac_thresh=3, filter_method="ransac"
    )
    assert seen and all(v == 3 for v in seen), seen


def test_preprocess_applies_moving_geometry(client):
    sid = client.post(
        "/api/session",
        json={"image_path": "moving.ome.tif", "reference_path": "reference.ome.tif"},
    ).json()["session_id"]

    def png(side, geometry):
        import io

        from PIL import Image

        r = client.post(
            f"/api/preprocess/{sid}/{side}",
            json={
                # percentile stretch only, so it commutes exactly with a flip
                "processor": "fluorescence",
                "params": {},
                "geometry": geometry,
                "size": 128,
            },
        )
        assert r.status_code == 200
        return np.asarray(Image.open(io.BytesIO(r.content)))

    plain = png("image", {})
    flipped = png("image", {"flip_h": True})
    np.testing.assert_array_equal(flipped, plain[:, ::-1])
    rotated = png("image", {"rotate": "90"})
    np.testing.assert_array_equal(rotated, np.rot90(plain, k=-1))
    # the reference defines the output frame: geometry is ignored there
    np.testing.assert_array_equal(
        png("reference", {"flip_h": True}), png("reference", {})
    )


def test_frontend_assets_revalidate(client):
    """Browsers must not serve a stale app.js after the frontend changes."""
    r = client.get("/static/app.js")
    assert r.status_code == 200
    assert r.headers["cache-control"] == "no-cache"


def test_result_outputs_lists_and_serves_valis_files(client, tmp_path):
    """The result page lists valis's plots (grouped, overlaps first), the
    summary metrics, and serves files only from inside the job directory."""
    out = tmp_path / "job_out"
    reg = out / "registration"
    for sub, name in [
        ("masks", "a.png"),
        ("overlaps", "registration_rigid_overlap.png"),
        ("matches", "m.png"),
    ]:
        (reg / sub).mkdir(parents=True, exist_ok=True)
        (reg / sub / name).write_bytes(b"\x89PNG fake")
    (reg / "data").mkdir()
    (reg / "data" / "registration_summary.csv").write_text(
        "filename,from,to,rigid_D,shape\n"
        "/x/mov.tif,mov,ref,4.5,\"(1, 2)\"\n"
        "/x/ref.tif,ref,,,\"(1, 2)\"\n"
    )
    (tmp_path / "secret.txt").write_text("nope")
    webapp_app._jobs["outjob"] = {
        "state": "error", "stage": "", "progress": 0.0, "message": "",
        "result": None, "out_dir": str(out),
    }
    try:
        data = client.get("/api/result/outjob/outputs").json()
        assert data["out_dir"] == os.path.realpath(out)
        assert [g["group"] for g in data["plot_groups"]] == [
            "overlaps", "matches", "masks",
        ]
        assert data["summary"] == [
            {"from": "mov", "to": "ref", "rigid_D": "4.5"}
        ]
        paths = {f["path"] for f in data["files"]}
        assert "registration/data/registration_summary.csv" in paths

        r = client.get(
            "/api/result/outjob/file",
            params={"path": "registration/overlaps/registration_rigid_overlap.png"},
        )
        assert r.status_code == 200 and r.content == b"\x89PNG fake"
        assert client.get(
            "/api/result/outjob/file", params={"path": "../secret.txt"}
        ).status_code == 400
        assert client.get("/api/result/nojob/outputs").status_code == 404
    finally:
        webapp_app._jobs.pop("outjob", None)


def test_result_plane_serves_each_page_separately(client, tmp_path):
    """Each viewer layer must get its own page of aligned.ome.tif (the tile
    source plugin ignored planeIndex and showed page 0 twice)."""
    import pyvips

    out = tmp_path / "planejob"
    out.mkdir()
    # larger than one 256px tile so the served copy has pyramid levels
    h, w = 520, 600
    pages = [np.full((h, w), v, np.uint8) for v in (40, 200)]
    stacked = pyvips.Image.arrayjoin(
        [pyvips.Image.new_from_array(p).cast("uchar") for p in pages], across=1
    )
    stacked.set_type(pyvips.GValue.gint_type, "page-height", h)
    tif = out / "aligned.ome.tif"
    stacked.tiffsave(str(tif), tile=True, pyramid=True, subifd=True)
    webapp_app._jobs["planejob"] = {
        "state": "done", "stage": "done", "progress": 1.0, "message": "",
        "result": str(tif), "out_dir": str(out),
    }
    try:
        for i, expected in enumerate((40, 200)):
            r = client.get(f"/api/result/planejob/plane/{i}.tif")
            assert r.status_code == 200
            f = tmp_path / f"got_{i}.tif"
            f.write_bytes(r.content)
            v = pyvips.Image.new_from_file(str(f))
            assert (v.width, v.height) == (w, h)
            assert abs(v.avg() - expected) < 3
            assert v.get_n_pages() > 1  # top-level IFD pyramid, zoomable
        assert client.get("/api/result/planejob/plane/2.tif").status_code == 404
        # the viewer cache is not a valis output
        listing = client.get("/api/result/planejob/outputs").json()
        assert not any(".viewer" in f["path"] for f in listing["files"])
    finally:
        webapp_app._jobs.pop("planejob", None)
