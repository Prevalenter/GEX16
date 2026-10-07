# GEX16

[Homepage](https://prevalenter.github.io/gexpro/) | [Documentation](https://prevalenter.github.io/GEXPRO-docs/) | [简体中文](README.zh-CN.md)

[![Teleoperation demo](docs/media/teleop.gif)](docs/media/teleop.mp4)  [![Mouse operation demo](docs/media/mouse.gif)](docs/media/mouse.mp4)

## Install

Python 3.10+:

```sh
python -m pip install -e '.[visualization]'
```

Base installation (`pip install -e .`) provides the hardware drivers and ZMQ.
For teleoperation, install `.[retarget]` in a Linux/CUDA environment with a suitable CUDA
PyTorch build. Real-device teleoperation still requires CUDA. The offline
viewer uses `.[retarget-viewer]` and supports CPU without SAPIEN. Use `.[dev]` for tests.

## Hardware And Visualization

Run each command in a separate terminal, using your actual serial ports:

```sh
python src/ex16.py --port /dev/ttyUSB0
python src/gx16.py --port /dev/ttyUSB1
python src/viewer.py
```

Use `--serial-number` instead of `--port` if needed. Open the Viser address
printed by the viewer (default `127.0.0.1:8080`). Both models display **measured
hardware angles**, not simulated or predicted states.

Both bases have translation/rotation gizmos. In **Base poses**, use **Save bases**
to persist them, **Reload bases** to restore the saved file, or **Reset bases**
to reset the scene without saving. Numeric position/RPY fields are synchronized
with the gizmos. **Gizmos** hides only the controls, not the models.
The default file is `~/.local/share/gex16/base_poses.json`; override with
`--base-poses /path/to/bases.json`. It loads automatically at startup. These
transforms affect visualization only, not robot calibration or hardware commands.

**GX16 initialization enables torque.** Verify wiring, motor IDs, directions,
mechanical zero and current/PWM settings before connecting. Keep a physical
power cutoff available.

## Retargeting

The **GEX Retarget** model and matching example inputs are bundled in:

```text
src/resources/retarget/
  last.pth       # Trained IK weights
  config.json    # Matching joint order, keypoint mapping and limits
  calibration.npz # EX16 calibration and original joint samples
  reference.npy   # Matching hand-keypoint reference data
```

Both entrypoints use these bundled files by default, including after wheel installation.
No external data paths are needed for the example. Override `--checkpoint-dir`,
`--calibration` or `--reference` to use different matching inputs. The included
calibration is for the existing example; check calibration suitability before
using a different hardware setup.

### Live Teleoperation

Start the EX16 publisher first, then use the bundled model and example data:

```sh
python src/teleop.py --checkpoint-dir src/resources/retarget \
  --calibration src/resources/retarget/calibration.npz \
  --reference src/resources/retarget/reference.npy
```

The default is Viser preview only. To command hardware, start `gex-gx16` and add
`--enable-gx16-output`. Use a different `--port-viser` if `gex-view` is also running.
Calibration loading is retained because inference needs it; calibration capture
and data generation are not included.

### Viser Teleoperation Test

No device or ZMQ publisher is needed. Run from the repository root:

```sh
python -m pip install -e '.[retarget-viewer]'
python src/retarget_viewer.py
```

The equivalent command with explicit bundled paths is:

```sh
python src/retarget_viewer.py \
  --checkpoint-dir src/resources/retarget \
  --calibration src/resources/retarget/calibration.npz \
  --reference src/resources/retarget/reference.npy \
  --device auto \
  --port-viser 8081
```

Open `127.0.0.1:8081`. Sixteen degree sliders, grouped by finger, drive EX16;
GX16 displays the corresponding GEX Retarget result, clipped to the checkpoint limits.
Slider ranges come from the EX16 URDF. **Zero joints** and **Calibration pose**
reset the inputs. This viewer shares the base gizmos and pose file with `viewer.py`.
Use a different `--base-poses` file for an independent layout.

**Save initial joints** saves all 16 current slider angles as the startup pose;
**Reload initial joints** restores it without restarting. The default file is
`~/.local/share/gex16/ex16_initial_pose.json`, configurable with `--initial-pose`.
On startup, a valid saved pose takes precedence over the calibration pose. A
missing/invalid file falls back to calibration (invalid files show an error).
Saving joints does not save the bases: use **Save bases** separately. Neither
setting modifies calibration, model weights or hardware.

`--device auto` selects CUDA when available, otherwise CPU; `--device cpu`
forces CPU inference. This tool has no hardware-output option, smoothing or
collision simulation. It tests the model mapping, not physical motion safety.

## Code Structure

```text
src/
  ex16.py          # Glove16 + EX16 publisher
  gx16.py          # Hand16 + GX16 command/state server
  viewer.py        # Live hardware visualization
  teleop.py        # GEX Retarget loop + Viser preview
  retarget_viewer.py # Offline EX16 sliders -> retargeting -> GX16
  device.py        # Serial setup/cleanup
  motor.py         # Required motor register operations
  transport.py     # ZMQ messages and GX16Client
  retargeting/     # Projection, model loading, smoothing
  resources/       # URDF, meshes, model and example calibration/reference data
  _vendor/         # Required third-party SDK and inference code
```

Python API: `from ex16 import Glove16` and `from gx16 import Hand16`.
`Glove16.getjs()` uses URDF degrees; `Hand16.getjs/setjs()` use motor-relative
degrees. Both drivers expose `connect()` and `close()`; always close hardware
in `finally`. Use `transport.GX16Client` for ZMQ commands (`request`),
or `get_qpos/set_qpos` in URDF radians. Do not open a device in two processes.

Core code lives directly in `src/`, without an extra package directory. Documentation,
tests and project configuration stay at the repository root. Run `python src/<module>.py`
from the repository root, or use `python -m <module>` after installation.
The installed commands `gex-ex16`, `gex-gx16`, `gex-view`, `gex-teleop`, and `gex-retarget-view` are
available. All support `--help`. Resources resolve relative to the code, not the
working directory. [Protocol and input files](docs/protocol.md) / [中文](docs/protocol.zh-CN.md).
Run `python -m pytest` for tests with injected test devices, never real hardware.
Live hardware and full Linux/CUDA/SAPIEN validation remain separate.

## Acknowledgments

We thank GeoRT for providing a foundation for the development of our retargeting system.
