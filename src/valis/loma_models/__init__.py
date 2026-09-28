"""Vendored inference code for LoMa (DaD detector + DeDoDe-G descriptor +
LoMa matcher).

Copied from https://github.com/davnords/LoMa (commit 8fb59c4), keeping only
the modules needed for inference, with ``loma.*`` imports made relative. It is
vendored rather than installed from PyPI (``lomatch``) because that package
depends on ``opencv-python``, which clobbers valis's
``opencv-contrib-python-headless`` (both ship the ``cv2`` module).

License: MIT (see LICENSE), except the matcher in ``loma.py``, which inherits
LightGlue's Apache-2.0 license.

Citation
--------
David Nordström, Johan Edstedt, et al. LoMa: Local Feature Matching Revisited.
ECCV 2026. https://arxiv.org/abs/2604.04931
"""

from .loma import LoMa as LoMa, LoMaB as LoMaB, filter_matches as filter_matches
