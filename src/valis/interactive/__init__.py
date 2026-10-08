"""Shared, importable building blocks for two-image alignment.

These modules are used by both the ``valis-align`` / ``valis-match`` CLIs
(:mod:`valis.cli`) and the
interactive web app (``valis.webapp``). Everything here imports ``valis`` (and
never torch directly) at module top so the valis-before-torch import ordering is
preserved for downstream callers.
"""

from valis.interactive import processors, pipeline  # noqa: F401
