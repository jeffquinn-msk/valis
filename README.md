See documentation for original project at `ReadTheDocs <https://valis.readthedocs.io/en/latest/>`_.

See `examples` for example usage of this fork.

## Changes in this fork

- All bioformats/java related dependencies removed. These complicated the project and I don't care about them. This fork only accepts TIFF files (this includes .OME.TIF files)
- Handling of .ome.tif inputs and multiple channels simplified (made opinioned decisions so it "just works" for my use case)
- Fixed a bug causing program to crash in single cpu environments
- Organized into a better python package structure so this can be used as a dependency in other python projects
- Containerization

## Known Issues

Python will segfault is this project (`valis`) is not imported first before any other pytorch-related import.
Don't ask me why!

License
-------

`MIT`_ © 2021-2025 Chandler Gatenbee
