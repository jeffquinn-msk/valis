"""FastAPI app for the interactive alignment web app.

Import ordering matters: ``valis`` (via ``valis.interactive``) is imported at
module top, before anything pulls in torch, to avoid the exit-139 segfault.
"""

import csv
import io
import os
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyvips
from PIL import Image as PILImage

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from valis.interactive import processors, pipeline
from valis.webapp import browse, matching

# ---------------------------------------------------------------------------
# Configuration + global state
# ---------------------------------------------------------------------------

DATA_ROOT = os.environ.get("VALIS_WEBAPP_DATA_ROOT", os.getcwd())
WORK_ROOT = os.environ.get(
    "VALIS_WEBAPP_WORK_ROOT", os.path.join(tempfile.gettempdir(), "valis_webapp")
)
DEFAULT_SIZE = 1024

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(title="Valis interactive alignment")


@app.middleware("http")
async def _revalidate_frontend(request, call_next):
    """Make browsers revalidate the page and its JS/CSS on every load (cheap:
    unchanged files come back 304 via ETag). Without this they heuristically
    cache a stale app.js after the frontend changes.
    """
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response

_sessions = {}  # session_id -> _Session
_jobs = {}  # job_id -> dict(state, stage, progress, message, result)
_jobs_lock = threading.Lock()
# Single worker: full registration is CPU/GPU-heavy; serialize runs.
_executor = ThreadPoolExecutor(max_workers=1)


class _Session:
    def __init__(self, image_path, reference_path):
        self.paths = {"image": image_path, "reference": reference_path}
        self._vips = {}  # side -> pyvips.Image
        self._thumbs = {}  # (side, size) -> {"gray": arr, "rgb": arr}
        self._lock = threading.Lock()

    def info(self, side):
        v = self._vips_for(side)
        return {"w": v.width, "h": v.height, "bands": v.bands}

    def _vips_for(self, side):
        v = self._vips.get(side)
        if v is None:
            v = pyvips.Image.new_from_file(self.paths[side], page=0)
            self._vips[side] = v
        return v

    def thumbs(self, side, size):
        key = (side, int(size))
        with self._lock:
            t = self._thumbs.get(key)
            if t is None:
                v = self._vips_for(side)
                t = {
                    "gray": processors.pyvips_to_thumbnail_array(v, int(size)),
                    "rgb": processors.pyvips_to_thumbnail_rgb_array(v, int(size)),
                }
                self._thumbs[key] = t
            return t


def _get_session(session_id) -> _Session:
    s = _sessions.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="unknown session")
    return s


def _png_response(arr: np.ndarray) -> Response:
    """Encode a uint8 numpy array (2-D gray or 3-D RGB) as a PNG response."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


def _thumb_for_processor(session: _Session, side: str, processor: str, size: int):
    """Return the raw thumbnail array (gray or rgb) a processor consumes."""
    kind = processors.PROCESSOR_INPUT.get(processor, "gray")
    return session.thumbs(side, size)[kind]


def _matcher_kwargs(matcher_cfg):
    """Normalize the frontend's matcher controls into ``build_matcher`` kwargs."""
    cfg = matcher_cfg or {}
    return {
        "matcher": cfg.get("matcher", "lightglue"),
        "detector": cfg.get("detector", "disk"),
        "max_keypoints": int(cfg.get("max_keypoints", 7500)),
        "ransac_thresh": float(cfg.get("ransac_thresh", 7)),
        "filter_method": cfg.get("filter_method", "magsac"),
    }


def _side_geometry(side, cfg):
    """Geometric pre-transform for ``side``. Only the moving image may be
    flipped/shifted; the reference defines the output frame, so any geometry
    sent for it is ignored.
    """
    if side != "image":
        return None
    return (cfg or {}).get("geometry")


def _run_processor(session, side, processor, params, size, geometry=None):
    if processor not in processors.PROCESSOR_REGISTRY:
        raise HTTPException(status_code=400, detail=f"unknown processor {processor!r}")
    cls, base_kw = processors.PROCESSOR_REGISTRY[processor]
    kwargs = {**base_kw, **(params or {})}
    thumb = _thumb_for_processor(session, side, processor, size)
    if not processors.geometry_is_identity(geometry):
        thumb = processors.apply_geometry_array(thumb, geometry)
    return processors.run_processor_on_thumbnail(
        [cls, kwargs], thumb, session.paths[side]
    )


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.get("/api/browse")
def api_browse(path: str = Query("")):
    try:
        return browse.list_dir(DATA_ROOT, path)
    except browse.BrowseError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/schema")
def api_schema():
    return processors.public_schema()


@app.post("/api/session")
def api_session(payload: dict = Body(...)):
    try:
        image_path = browse.resolve_within_root(DATA_ROOT, payload["image_path"])
        reference_path = browse.resolve_within_root(
            DATA_ROOT, payload["reference_path"]
        )
    except (KeyError, browse.BrowseError) as e:
        raise HTTPException(status_code=400, detail=f"bad path: {e}")
    for p in (image_path, reference_path):
        if not os.path.isfile(p):
            raise HTTPException(status_code=400, detail=f"not a file: {p}")

    session_id = uuid.uuid4().hex
    session = _Session(image_path, reference_path)
    _sessions[session_id] = session
    # Suggested defaults from the auto-stain resolver.
    try:
        image_stain = processors.resolve_auto_stain(image_path)
        reference_stain = processors.resolve_auto_stain(reference_path)
    except Exception:
        image_stain = reference_stain = None
    return {
        "session_id": session_id,
        "image": {**session.info("image"), "suggested_stain": image_stain},
        "reference": {**session.info("reference"), "suggested_stain": reference_stain},
    }


@app.get("/api/thumbnail/{session_id}/{side}")
def api_thumbnail(session_id: str, side: str, size: int = Query(DEFAULT_SIZE)):
    if side not in ("image", "reference"):
        raise HTTPException(status_code=400, detail="side must be image|reference")
    session = _get_session(session_id)
    v = session._vips_for(side)
    # Show RGB for color images, gray for single band.
    kind = "rgb" if v.bands >= 3 else "gray"
    return _png_response(session.thumbs(side, size)[kind])


@app.post("/api/preprocess/{session_id}/{side}")
def api_preprocess(session_id: str, side: str, payload: dict = Body(...)):
    if side not in ("image", "reference"):
        raise HTTPException(status_code=400, detail="side must be image|reference")
    session = _get_session(session_id)
    size = int(payload.get("size", DEFAULT_SIZE))
    processor = payload.get("processor")
    params = payload.get("params", {})
    geometry = _side_geometry(side, payload)
    try:
        out = _run_processor(session, side, processor, params, size, geometry)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"preprocess failed: {e}")
    return _png_response(out)


@app.post("/api/match/{session_id}")
def api_match(session_id: str, payload: dict = Body(...)):
    session = _get_session(session_id)
    size = int(payload.get("size", DEFAULT_SIZE))
    img_cfg = payload.get("image", {})
    ref_cfg = payload.get("reference", {})
    try:
        img_proc = _run_processor(
            session,
            "image",
            img_cfg.get("processor"),
            img_cfg.get("params", {}),
            size,
            _side_geometry("image", img_cfg),
        )
        ref_proc = _run_processor(
            session,
            "reference",
            ref_cfg.get("processor"),
            ref_cfg.get("params", {}),
            size,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"preprocess failed: {e}")

    try:
        kp1, kp2, n_total, n_filtered = matching.detect_and_match(
            img_proc, ref_proc, **_matcher_kwargs(payload.get("matcher"))
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"matching failed: {e}")

    return {
        "matches": {"kp1": kp1.tolist(), "kp2": kp2.tolist()},
        "n_total": n_total,
        "n_filtered": n_filtered,
        "image_size": [img_proc.shape[1], img_proc.shape[0]],
        "reference_size": [ref_proc.shape[1], ref_proc.shape[0]],
    }


def _run_alignment_job(
    job_id, session, img_cfg, ref_cfg, matcher_cfg, max_dim, min_matches
):
    with _jobs_lock:
        out_dir = _jobs[job_id].get("out_dir") or os.path.join(WORK_ROOT, job_id)
    os.makedirs(out_dir, exist_ok=True)

    def progress_cb(stage, frac, msg=""):
        with _jobs_lock:
            j = _jobs[job_id]
            j["stage"] = stage
            j["progress"] = float(frac)
            j["message"] = msg

    with _jobs_lock:
        _jobs[job_id].update(state="running", stage="starting", progress=0.0)
    try:
        aligned = pipeline.run_alignment(
            image_path=session.paths["image"],
            reference_path=session.paths["reference"],
            output_dir=out_dir,
            image_stain=img_cfg.get("processor", "auto"),
            reference_stain=ref_cfg.get("processor", "auto"),
            image_params=img_cfg.get("params", {}),
            image_geometry=_side_geometry("image", img_cfg),
            reference_params=ref_cfg.get("params", {}),
            max_processed_image_dim_px=int(max_dim),
            min_rigid_matches=int(min_matches),
            matcher_cfg=matcher_cfg,
            progress_cb=progress_cb,
        )
        with _jobs_lock:
            _jobs[job_id].update(
                state="done",
                stage="done",
                progress=1.0,
                message="alignment complete",
                result=aligned,
            )
    except pipeline.AlignmentError as e:
        with _jobs_lock:
            _jobs[job_id].update(state="error", message=str(e))
    except Exception as e:  # noqa: BLE001 - surface any failure to the UI
        with _jobs_lock:
            _jobs[job_id].update(state="error", message=f"{type(e).__name__}: {e}")


@app.post("/api/align/{session_id}")
def api_align(session_id: str, payload: dict = Body(...)):
    session = _get_session(session_id)
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "state": "queued",
            "stage": "queued",
            "progress": 0.0,
            "message": "",
            "result": None,
            "out_dir": os.path.join(WORK_ROOT, job_id),
        }
    _executor.submit(
        _run_alignment_job,
        job_id,
        session,
        payload.get("image", {}),
        payload.get("reference", {}),
        _matcher_kwargs(payload.get("matcher")),
        payload.get("max_dim", 2048),
        payload.get("min_matches", 30),
    )
    return {"job_id": job_id}


@app.get("/api/align/{job_id}/status")
def api_align_status(job_id: str):
    with _jobs_lock:
        j = _jobs.get(job_id)
        if j is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return {
            "state": j["state"],
            "stage": j["stage"],
            "progress": j["progress"],
            "message": j["message"],
            "result_ready": j["state"] == "done",
        }


@app.get("/api/result/{job_id}/aligned.ome.tif")
def api_result(job_id: str):
    with _jobs_lock:
        j = _jobs.get(job_id)
    if j is None or j.get("result") is None:
        raise HTTPException(status_code=404, detail="result not ready")
    path = j["result"]
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="result file missing")
    # FileResponse handles HTTP Range requests, which GeoTIFFTileSource needs.
    return FileResponse(path, media_type="image/tiff", filename="aligned.ome.tif")


# Viewer copies live in a hidden subfolder so they stay out of the output
# listing (they're a display cache, not a valis output).
VIEWER_DIR = ".viewer"
_plane_locks = {}  # job_id -> Lock (so concurrent tile requests write once)


@app.get("/api/result/{job_id}/plane/{index}.tif")
def api_result_plane(job_id: str, index: int):
    """Serve one page of ``aligned.ome.tif`` as its own pyramidal TIFF.

    The browser's GeoTIFFTileSource can't pick a page out of our output: it
    doesn't read SubIFD pyramids, and its fallback ignores
    ``hints.layout.planeIndex``, so every layer showed page 0. A single-page
    TIFF with a top-level (IFD) pyramid is both selectable and zoomable.
    Written on first request, then cached next to the job's output.
    """
    with _jobs_lock:
        j = _jobs.get(job_id)
        if j is not None:
            lock = _plane_locks.setdefault(job_id, threading.Lock())
    if j is None or j.get("result") is None or not os.path.isfile(j["result"]):
        raise HTTPException(status_code=404, detail="result not ready")
    src = j["result"]
    n_pages = pyvips.Image.new_from_file(src).get_n_pages()
    if not 0 <= index < n_pages:
        raise HTTPException(status_code=404, detail=f"no plane {index}")
    out_dir = os.path.join(os.path.dirname(src), VIEWER_DIR)
    path = os.path.join(out_dir, f"plane_{index}.tif")
    with lock:
        if not os.path.isfile(path):
            os.makedirs(out_dir, exist_ok=True)
            tmp = path + ".tmp"
            pyvips.Image.new_from_file(src, page=index).tiffsave(
                tmp,
                tile=True,
                tile_width=256,
                tile_height=256,
                pyramid=True,
                compression="jpeg",
                Q=90,
                bigtiff=True,
            )
            os.replace(tmp, path)
    return FileResponse(path, media_type="image/tiff")


# valis output subfolders in the order the result page shows them: the
# overlaps answer "did it work?", the rest explain how it got there.
PLOT_GROUP_ORDER = (
    "overlaps",
    "matches",
    "rigid_registration",
    "non_rigid_registration",
    "deformation_fields",
    "processed",
    "masks",
)
PLOT_EXTS = (".png", ".jpg", ".jpeg")
# Summary CSV columns worth showing (the rest are shapes / paths).
SUMMARY_COLUMNS = (
    "from",
    "to",
    "original_D",
    "rigid_D",
    "non_rigid_D",
    "original_rTRE",
    "rigid_rTRE",
    "non_rigid_rTRE",
    "physical_units",
    "rigid_time_minutes",
    "non_rigid_time_minutes",
)


def _job_out_dir(job_id: str) -> str:
    with _jobs_lock:
        j = _jobs.get(job_id)
    if j is None:
        raise HTTPException(status_code=404, detail="unknown job")
    out_dir = j.get("out_dir")
    if not out_dir or not os.path.isdir(out_dir):
        raise HTTPException(status_code=404, detail="no output directory yet")
    return out_dir


def _read_summary(out_dir: str):
    """Rows of valis's registration summary CSV for the moving slides."""
    for root, _, names in os.walk(out_dir):
        for name in names:
            if name.endswith("_summary.csv"):
                with open(os.path.join(root, name), newline="") as f:
                    rows = [r for r in csv.DictReader(f) if r.get("to")]
                return [
                    {c: r.get(c, "") for c in SUMMARY_COLUMNS if c in r} for r in rows
                ]
    return []


@app.get("/api/result/{job_id}/outputs")
def api_result_outputs(job_id: str):
    """Everything valis wrote for a job: its directory, the plots grouped by
    subfolder, the summary metrics, and a flat file listing. Available once
    the job has started, so a failed run's diagnostic plots show up too.
    """
    out_dir = _job_out_dir(job_id)
    files, plots = [], {}
    for root, dirs, names in os.walk(out_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(names):
            full = os.path.join(root, name)
            rel = os.path.relpath(full, out_dir)
            is_link = os.path.islink(full)
            try:
                size = os.path.getsize(full)
            except OSError:
                size = None
            files.append({"path": rel, "size": size, "link": is_link})
            if not is_link and name.lower().endswith(PLOT_EXTS):
                group = os.path.basename(root)
                plots.setdefault(group, []).append({"name": name, "path": rel})
    order = {g: i for i, g in enumerate(PLOT_GROUP_ORDER)}
    groups = sorted(plots, key=lambda g: (order.get(g, len(order)), g))
    return {
        "out_dir": os.path.realpath(out_dir),
        "plot_groups": [{"group": g, "plots": plots[g]} for g in groups],
        "summary": _read_summary(out_dir),
        "files": files,
    }


@app.get("/api/result/{job_id}/file")
def api_result_file(job_id: str, path: str = Query(...)):
    """Serve one file from a job's output directory (sandboxed to it)."""
    out_dir = _job_out_dir(job_id)
    try:
        full = browse.resolve_within_root(out_dir, path)
    except browse.BrowseError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not os.path.isfile(full):
        raise HTTPException(status_code=404, detail="no such file")
    return FileResponse(full, filename=os.path.basename(full))


# ---------------------------------------------------------------------------
# Static frontend (mounted last so /api/* wins)
# ---------------------------------------------------------------------------


@app.get("/")
def index():
    index_html = os.path.join(STATIC_DIR, "index.html")
    if not os.path.isfile(index_html):
        return JSONResponse({"detail": "frontend not built"}, status_code=404)
    return FileResponse(index_html)


if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# geotiff-tilesource loads its tile-decoding worker (and the worker's codec
# chunks) from the absolute path /assets/..., so serve its dist/assets there.
_PLUGIN_ASSETS_DIR = os.path.join(STATIC_DIR, "vendor", "assets")
if os.path.isdir(_PLUGIN_ASSETS_DIR):
    app.mount("/assets", StaticFiles(directory=_PLUGIN_ASSETS_DIR), name="assets")
