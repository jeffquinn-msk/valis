"""FastAPI app for the interactive alignment web app.

Import ordering matters: ``valis`` (via ``valis.interactive``) is imported at
module top, before anything pulls in torch, to avoid the exit-139 segfault.
"""

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


def _run_processor(session, side, processor, params, size):
    if processor not in processors.PROCESSOR_REGISTRY:
        raise HTTPException(status_code=400, detail=f"unknown processor {processor!r}")
    cls, base_kw = processors.PROCESSOR_REGISTRY[processor]
    kwargs = {**base_kw, **(params or {})}
    thumb = _thumb_for_processor(session, side, processor, size)
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
    try:
        out = _run_processor(session, side, processor, params, size)
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
    out_dir = os.path.join(WORK_ROOT, job_id)
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
