"""Sandboxed directory browsing rooted at a configured data directory.

All user-supplied paths are resolved and checked to be inside the root, so the
browser can never escape (``..``, absolute paths, symlinks pointing out).
"""

import os

# Image extensions the browser will surface as selectable.
IMAGE_EXTS = (".ome.tif", ".ome.tiff", ".tif", ".tiff", ".svs", ".ndpi", ".czi")


class BrowseError(ValueError):
    """Raised for an out-of-sandbox or invalid path."""


def _is_image(name: str) -> bool:
    lower = name.lower()
    return any(lower.endswith(ext) for ext in IMAGE_EXTS)


def resolve_within_root(root: str, rel_path: str) -> str:
    """Resolve ``rel_path`` (relative to ``root``) and ensure it stays inside
    ``root``. Returns the absolute path. Raises :class:`BrowseError` on escape.
    """
    root_abs = os.path.realpath(root)
    candidate = os.path.realpath(os.path.join(root_abs, rel_path or ""))
    if candidate != root_abs and not candidate.startswith(root_abs + os.sep):
        raise BrowseError(f"path escapes data root: {rel_path!r}")
    return candidate


def list_dir(root: str, rel_path: str = "") -> dict:
    """List directories and image files under ``root/rel_path``.

    Returns ``{"path": rel, "parent": rel_parent_or_None,
    "dirs": [...], "files": [{"name","path"}]}`` with ``path`` values relative
    to ``root`` (so the frontend never sees absolute host paths).
    """
    abs_path = resolve_within_root(root, rel_path)
    if not os.path.isdir(abs_path):
        raise BrowseError(f"not a directory: {rel_path!r}")

    root_abs = os.path.realpath(root)
    rel = os.path.relpath(abs_path, root_abs)
    rel = "" if rel == "." else rel

    dirs, files = [], []
    with os.scandir(abs_path) as it:
        for entry in it:
            if entry.name.startswith("."):
                continue
            entry_rel = os.path.relpath(entry.path, root_abs)
            try:
                if entry.is_dir():
                    dirs.append({"name": entry.name, "path": entry_rel})
                elif entry.is_file() and _is_image(entry.name):
                    files.append({"name": entry.name, "path": entry_rel})
            except OSError:
                continue

    dirs.sort(key=lambda d: d["name"].lower())
    files.sort(key=lambda f: f["name"].lower())

    parent = None
    if rel:
        parent = os.path.dirname(rel)
        parent = parent if parent else ""

    return {"path": rel, "parent": parent, "dirs": dirs, "files": files}
