"""Cached DISK/DeDoDe/LoMa/RoMa v2 detectors + matchers and the fast
detect-and-match service used by the ``/api/match`` endpoint.

Loading detector weights is the expensive part, so detectors and matchers are
cached by configuration and kept resident across requests. torch models are not
thread-safe, so all inference is serialized behind a single lock.
"""

import threading

import numpy as np

from valis.interactive import pipeline

_lock = threading.Lock()
_detectors = {}  # (matcher, detector_type, max_keypoints) -> FeatureDD
_matchers = {}  # (det_key, filter_method, ransac_thresh) -> Matcher


def _get_detector(detector: str, max_keypoints: int, matcher: str = "lightglue"):
    # LoMa and RoMa v2 ignore ``detector``; don't load their weights once per
    # detector name.
    if matcher in ("loma-b", "romav2"):
        detector = None
    # For RoMa v2, max_keypoints is only how many matches to sample: keep one
    # model (~1.1 GB) and update the count instead of loading another.
    key = (matcher, detector, None if matcher == "romav2" else int(max_keypoints))
    det = _detectors.get(key)
    if det is None:
        det = pipeline.build_detector(detector, max_keypoints, matcher)
        _detectors[key] = det
    if matcher == "romav2":
        det.num_features = int(max_keypoints)
    return det, key


def _get_matcher(det, det_key, filter_method: str, ransac_thresh: float):
    key = (det_key, filter_method, int(ransac_thresh))
    mat = _matchers.get(key)
    if mat is None:
        mat = pipeline.build_matcher(
            filter_method=filter_method,
            ransac_thresh=ransac_thresh,
            matcher=det_key[0],
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
    matcher: str = "lightglue",
):
    """Detect keypoints on both preprocessed thumbnails and match with
    LightGlue or LoMa, or match them densely with RoMa v2.

    Returns ``(matched_kp1_xy, matched_kp2_xy, n_total, n_filtered)`` where the
    kp arrays are the matched coordinate pairs left after ``filter_method``
    (all of them for ``"none"``), in thumbnail pixel space.

    Keypoints are pre-detected and passed explicitly so ``match_images`` takes
    its no-internal-rotation path.
    """
    img1_u8 = np.ascontiguousarray(img1_u8)
    img2_u8 = np.ascontiguousarray(img2_u8)

    with _lock:
        det, det_key = _get_detector(detector, max_keypoints, matcher)
        mat = _get_matcher(det, det_key, filter_method, ransac_thresh)

        kp1, d1 = det.detect_and_compute(img1_u8)
        kp2, d2 = det.detect_and_compute(img2_u8)

        match12, filt12, _, _ = mat.match_images(
            img1_u8, img2_u8, desc1=d1, kp1_xy=kp1, desc2=d2, kp2_xy=kp2
        )

    # The matcher's own filtered output: the same filter valis's rigid step
    # runs (with "none" it keeps every match).
    n_total = int(len(match12.matched_kp1_xy))
    n_filtered = int(len(filt12.matched_kp1_xy))
    return (
        np.asarray(filt12.matched_kp1_xy, dtype=float),
        np.asarray(filt12.matched_kp2_xy, dtype=float),
        n_total,
        n_filtered,
    )
