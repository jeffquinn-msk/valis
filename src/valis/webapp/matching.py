"""Cached DISK/DeDoDe detectors + LightGlue matchers and the fast
detect-and-match service used by the ``/api/match`` endpoint.

Loading detector weights is the expensive part, so detectors and matchers are
cached by configuration and kept resident across requests. torch models are not
thread-safe, so all inference is serialized behind a single lock.
"""

import threading

import numpy as np

from valis.interactive import pipeline

_lock = threading.Lock()
_detectors = {}  # (detector_type, max_keypoints) -> FeatureDD
_matchers = {}  # (det_key, filter_method, ransac_thresh) -> LightGlueMatcher


def _get_detector(detector: str, max_keypoints: int):
    key = (detector, int(max_keypoints))
    det = _detectors.get(key)
    if det is None:
        det = pipeline.build_detector(detector, max_keypoints)
        _detectors[key] = det
    return det, key


def _get_matcher(det, det_key, filter_method: str, ransac_thresh: float):
    key = (det_key, filter_method, int(ransac_thresh))
    mat = _matchers.get(key)
    if mat is None:
        mat = pipeline.build_matcher(
            filter_method=filter_method,
            ransac_thresh=ransac_thresh,
            feature_detector=det,
        )
        _matchers[key] = mat
    return mat


def detect_and_match(
    img1_u8: np.ndarray,
    img2_u8: np.ndarray,
    detector: str = "disk",
    max_keypoints: int = 7500,
    ransac_thresh: float = 7,
    filter_method: str = "magsac",
):
    """Detect keypoints on both preprocessed thumbnails and match with LightGlue.

    Returns ``(matched_kp1_xy, matched_kp2_xy, n_total, n_filtered)`` where the
    kp arrays are the (filtered, unless ``filter_method == "none"``) matched
    coordinate pairs in thumbnail pixel space.

    Keypoints are pre-detected and passed explicitly so ``match_images`` takes
    its no-internal-rotation path (the two-arg form crashes on ``kp2_xy=None``).
    """
    img1_u8 = np.ascontiguousarray(img1_u8)
    img2_u8 = np.ascontiguousarray(img2_u8)

    with _lock:
        det, det_key = _get_detector(detector, max_keypoints)
        mat = _get_matcher(det, det_key, filter_method, ransac_thresh)

        kp1, d1 = det.detect_and_compute(img1_u8)
        kp2, d2 = det.detect_and_compute(img2_u8)

        match12, filt12, _, _ = mat.match_images(
            img1_u8, img2_u8, desc1=d1, kp1_xy=kp1, desc2=d2, kp2_xy=kp2
        )

    n_total = int(len(match12.matched_kp1_xy))
    if filter_method == "none":
        chosen = match12
    else:
        chosen = filt12
    n_filtered = int(len(chosen.matched_kp1_xy))
    return (
        np.asarray(chosen.matched_kp1_xy, dtype=float),
        np.asarray(chosen.matched_kp2_xy, dtype=float),
        n_total,
        n_filtered,
    )
