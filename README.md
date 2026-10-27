# virtual-camera

A small software 3D renderer written in Python. It loads OBJ models (or a TOML scene that places several of them), projects them through a movable perspective camera, and draws them with pygame-ce. Hidden surfaces are resolved with a BSP tree. Every transform and projection is done by hand with NumPy.

## Requirements

- Python 3.14
- [uv](https://docs.astral.sh/uv/)

## Setup

```sh
uv sync
```

This creates `.venv/`, installs the dependencies from `uv.lock`, and installs the project itself in editable mode, so the `virtual-camera` command is available.

## Running

```sh
uv run virtual-camera                  # loads the built-in example scene
uv run virtual-camera model.obj        # any mix of .obj and .toml files
uv run virtual-camera -v               # debug logging
uv run virtual-camera -h               # all options
```

Options:

| Flag | Meaning | Default |
|---|---|---|
| `-s`, `--screenshot-dir` | Where F12 saves screenshots | `screenshots/` in the current directory |
| `-m`, `--move-speed` | Units per second | 6.0 |
| `-r`, `--rot-speed` | Radians per second | 3.0 |
| `-v`, `--verbose` | Show debug output | off |

## Controls

| Key | Action |
|---|---|
| W / S | Move forward / back |
| A / D | Move left / right |
| Q / E | Move up / down |
| Arrow keys | Look up, down, left, right |
| Z / C | Roll left / right |
| + / - or mouse wheel | Zoom in / out (FOV) |
| R | Reset camera |
| F12 | Save screenshot |
| Esc | Quit |

## Scene files

A scene is a TOML file with one `[[model]]` table per object. `file` is resolved relative to the scene file.

```toml
[[model]]
file = "../solids/cube.obj"
pos = [10, 0, 0]     # translation
rot = [0, 45, 0]     # rotation in degrees (x, y, z)
scale = [1, 1, 1]
```

## Project layout

```
src/virtual_camera/
  __main__.py   entry point and render loop
  config.py     window size, speeds, default paths
  app/          input handling, HUD, screenshots
  render/       camera, transforms, BSP tree, renderer
  scene/        OBJ/TOML loading and model data
  assets/       built-in sample models and the default scene,
                shipped with the package
    solids/     OBJ models
    scenes/     TOML scenes
```
