"""Entry point for ``valis-webapp``.

Imports valis (via the webapp app module, which imports ``valis.interactive``)
BEFORE uvicorn/torch, preserving the valis-before-torch ordering.
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Interactive alignment web app.")
    parser.add_argument(
        "--data-root",
        default=os.getcwd(),
        help="Directory the file browser is sandboxed to (default: cwd).",
    )
    parser.add_argument(
        "--work-root",
        default=None,
        help="Directory for alignment outputs (default: a temp dir).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    os.environ["VALIS_WEBAPP_DATA_ROOT"] = os.path.realpath(args.data_root)
    if args.work_root:
        os.environ["VALIS_WEBAPP_WORK_ROOT"] = os.path.realpath(args.work_root)

    # Import the app (pulls in valis) before uvicorn imports anything torch-y.
    from valis.webapp import app as _app  # noqa: F401
    import uvicorn

    print(f"[valis-webapp] data root: {os.environ['VALIS_WEBAPP_DATA_ROOT']}")
    print(f"[valis-webapp] serving on http://{args.host}:{args.port}")
    uvicorn.run(_app.app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
