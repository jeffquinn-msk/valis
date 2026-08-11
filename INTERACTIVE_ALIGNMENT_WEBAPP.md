# Problem Statement

Right now in `scripts/align_two_images.py` we have multiple preprocessor options. These different preprocessors also have different parameters. Users need to cycle through different options manually to see which one gives the best feature matching solution when
run through LightGlue.

The preprocessing and inference pass through LightGlue is a relatively quick operation.

Given these facts, it would be a much more productive user experience if there was an interactive web app where a user could tweak preprocessing parameters and see the LightGlue inference in real time.

Images can be downsampled and displayed in a `<canvas>` element. The lightglue features and connections can also be rendered in that canvas

There should be a button in the webapp to launch the full alignment, which takes longer. When the alignment is done, the user shuold be able to view it interactively as well with pan and zoom, utilizing 

These are large image files, youll need a way to downsample and send them to the frontend, see https://github.com/napari/napari/blob/main/README.md for an example of python project that does this

# UI

The ui should have the two downsampled images side by side, with a dropdown below each to select preprocessors, and parameter sliders
below that to choose params for the preprocessors

There shuold be a button called "Run Keypoint Detection" which will run LightGlue with the current settings

below that should be a button run alignment

