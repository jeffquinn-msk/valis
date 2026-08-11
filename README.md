See documentation for original project at https://valis.readthedocs.io/en/latest/.

See `examples` for example usage of this fork.

## Changes in this fork

- All bioformats/java related dependencies removed. These complicated the project and I don't care about them. This fork only accepts TIFF files (this includes .OME.TIF files)
- Handling of .ome.tif inputs and multiple channels simplified (made opinioned decisions so it "just works" for my use case)
- Fixed a bug causing program to crash in single cpu environments
- Organized into a better python package structure so this can be used as a dependency in other python projects
- Containerization

## Running the smoketest

A smoketest that downloads two small example images and runs a full registration is in `tests/test_align_two.py`. On first run it fetches ~12MB from the upstream repo and caches them locally; subsequent runs are offline.

```bash
.venv/bin/python -m pytest tests/test_align_two.py -v
```

After it passes, visual artifacts are written to `tests/test_output/smoketest/`:

- `overlaps/smoketest_original_overlap.png` — the two images before alignment
- `overlaps/smoketest_rigid_overlap.png` — after rigid registration
- `overlaps/smoketest_non_rigid_overlap.png` — after non-rigid registration
- `deformation_fields/` — warp meshes showing how much each image was corrected

## Interactive alignment web app

An interactive web app lets you tune preprocessing per image and see DISK+LightGlue
matches update in real time on downsampled thumbnails, then launch the full
registration and pan/zoom the aligned result in the browser.

```bash
uv sync --extra web
.venv/bin/valis-webapp --data-root /path/to/your/slides
# open http://127.0.0.1:8000
```

Workflow: **Open images…** browses `--data-root` (sandboxed) to pick a reference and a
moving `.ome.tif`; each panel has a preprocessor dropdown + parameter sliders and a
shared detector/matcher control block. **Run Keypoint Detection** overlays the live
LightGlue matches; **Run Alignment** runs the full pipeline and opens the aligned
`aligned.ome.tif` in an OpenSeadragon viewer (via the bundled GeoTIFFTileSource plugin,
which reads the pyramidal TIFF directly — no tile export).

The shared preprocessing/registration logic lives in `valis.interactive` (used by both
this app and `scripts/align_two_images.py`).

## Known Issues

Python will segfault is this project (`valis`) is not imported first before any other pytorch-related import.
Don't ask me why!

License
-------

`MIT` © 2021-2025 Chandler Gatenbee
