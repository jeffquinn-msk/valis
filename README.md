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

## Aligning two images from the command line

`scripts/align_two_images.py` registers a moving image to a reference image
(rigid + non-rigid) and writes the warped result as a pyramidal OME-TIFF.

```bash
.venv/bin/python scripts/align_two_images.py \
    --reference /path/to/reference.ome.tif \
    --image /path/to/moving.ome.tif \
    --output-dir /path/to/output
```

Only page 0 of each input is used (e.g. the DAPI channel of a multichannel
`.ome.tif`), and 16-bit inputs are converted to 8-bit.

### Outputs

In `--output-dir`:

- `aligned.ome.tif`: two-page OME-TIFF, page 0 is the moving image warped onto the
  reference, page 1 is the reference.
- `registration/`: valis's working files, including `matches/` (keypoint-match
  visualizations, also written for a failed attempt) and the processed images and masks.
- An 8-bit copy of the reference, and the moving image (a symlink, or a rotated/flipped
  copy if the orientation check corrected it).

### Options

| Flag | Default | Description |
|---|---|---|
| `--image-stain`, `--reference-stain` | `auto` | Preprocessor for each image: `he-hematoxylin`, `he-hematoxylin-raw`, `he-hematoxylin-sparse`, `fluorescence`, `inverted-fluorescence`, `od`, `colorful-standardizer`, `luminosity`. `auto` picks `he-hematoxylin` for RGB and `fluorescence` / `inverted-fluorescence` for single-band images, based on brightness. |
| `--max-processed-image-dim-px` | `2048` | Longest side used for feature detection and non-rigid registration. Never exceeds the images' own size. |
| `--min-rigid-matches` | `30` | Abort with an error if the first matching pass finds fewer matches than this. |
| `--orientation-margin` | `0.0` | Minimum score margin required before the orientation check rotates or flips the moving image. |
| `--no-script-orientation` | off | Skip the script's orientation check and let valis handle reflections. |
| `--matcher` | `lightglue` | Feature matcher: `lightglue`, `loma-b` or `romav2`. |
| `--detector` | `disk` | Feature detector used with LightGlue: `disk` or `dedode`. |
| `--max-keypoints` | `7500` | Maximum keypoints per image (with `romav2`, matches sampled per image pair). |
| `--ransac-thresh` | `7` | Outlier-filter reprojection threshold, in pixels. |
| `--filter-method` | `magsac` | Outlier filter for matches: `magsac` or `ransac`. |

`--matcher loma-b` swaps LightGlue for [LoMa-B](https://github.com/davnords/LoMa). LoMa
brings its own DaD keypoints and DeDoDe-G descriptors, so it ignores `--detector`. Its
inference code is vendored in `valis.loma_models`. The first run downloads about 2 GB of
weights (the LoMa-B checkpoint, DaD and DINOv2 ViT-L), and it needs roughly 4–5 GB of
RAM. In Python, use `feature_matcher.LoMaMatcher()`, which builds its own
`feature_detectors.LoMaFD`.

`--matcher romav2` uses [RoMa v2](https://github.com/Parskatt/romav2), a dense matcher
(installed from PyPI with the `dl` extra). It has no keypoint detector: it matches the
two images directly and samples `--max-keypoints` correspondences from the dense warp,
so it ignores `--detector`. It runs RoMa v2's `base` setting (640×640, one direction),
which uses about 7.5 GB of GPU memory on Apple silicon. The first run downloads about
1.1 GB of weights and fetches the DINOv3 backbone code from GitHub. In Python, use
`feature_matcher.RoMaV2Matcher()`, which builds its own `feature_detectors.RoMaV2FD`
(pass `RoMaV2FD(setting="precise")` for RoMa v2's more accurate, much heavier preset).

`--matcher`, `--detector`, `--max-keypoints`, `--ransac-thresh` and `--filter-method`
are the same controls as the web app's detector/matcher panel. If none is given,
valis's default matcher is used.

If `he-hematoxylin` gets too few matches, the script automatically retries with the
sparse hematoxylin extractor over several parameter settings before giving up.

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
LightGlue matches; **Run Alignment** runs the same pipeline as the CLI, using the
panel's settings, and opens `aligned.ome.tif` in an OpenSeadragon viewer. The viewer
shows the aligned moving image in green and the reference in magenta, so aligned
tissue appears white and misalignment appears as colored fringes. It uses the bundled
GeoTIFFTileSource plugin, which reads the pyramidal TIFF directly (no tile export).

The shared preprocessing/registration logic lives in `valis.interactive` (used by both
this app and `scripts/align_two_images.py`).

## Known Issues

Python will segfault is this project (`valis`) is not imported first before any other pytorch-related import.
Don't ask me why!

DISK feature detection needs a lot of memory on CPU, and the need grows steeply with
image size. A full alignment of the ~1800px example images (processed at ~2000px)
exhausted a 36GB machine. Start with a lower `--max-processed-image-dim-px`, or
downsampled copies of the images, when trying out a new pair.

License
-------

`MIT` © 2021-2025 Chandler Gatenbee
