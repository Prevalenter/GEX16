# Protocol And Retargeting Inputs

English | [简体中文](protocol.zh-CN.md)

## Hardware Messages

PUB messages are one UTF-8 frame: `<topic> <JSON object>`. Defaults are loopback.

| Publisher | Endpoint | Topic | Fields |
| --- | --- | --- | --- |
| EX16 | `tcp://127.0.0.1:5567` | `ex16/state` | `name`, `sequence`, `timestamp`, `urdf_deg` |
| GX16 | `tcp://127.0.0.1:5557` | `gx16/state` | `timestamp`, measured `motor_deg`, measured `urdf_deg` |

Angles are 16 finite degrees, ordered joint1..joint16: thumb, index, middle,
ring, four joints each. GX16 URDF degrees equal motor-relative degrees times
the configured direction signs. The hardware driver adds the 90-degree offset.
`gex-view` subscribes to these two streams directly. The old retargeted-state
topic and control-status telemetry have been removed.

GX16 REP binds `tcp://127.0.0.1:5556`. A request is `{ "cmd": "...", ... }`.
Replies use `{ "ok": boolean, "result": object|null, "error": string|null }`.

| Command | Parameters / result |
| --- | --- |
| `ping`, `status` | Returns `connected`, `port` |
| `getjs` | Returns `positions`, `units` |
| `setjs` | `positions`: 16 finite numbers |
| `setj` | `joint`: integer 1..16; `position`: finite number |
| `home` | Moves every motor to its zero-relative position |
| `torque_on`, `torque_off` | Enables/disables motor torque |
| `shutdown` | Exits the server and attempts torque-off |

`getjs/setjs/setj` accept `units="urdf_deg"` (default) or `"motor_deg"`.
Successful write commands return an empty result object. Direct motion commands
do not apply retargeting joint limits or collision checks. `teleop_start/stop`, gate
mode, dry-run and the separate client CLI are removed. Use `GX16Client.request()`
or the driver API. The server uses one loop for serial I/O and ZMQ, so slow
hardware reads also delay command handling; `--state-hz` controls read frequency.

## GEX Retarget Inputs

- `--checkpoint-dir`: defaults to the bundled `src/resources/retarget` directory
  (`resources/retarget` inside an installed package). It contains `last.pth`
  and matching `config.json`, including
  `joint_order`, `fingertip_link` mappings and `joint.lower/upper` limits.
- `--calibration`: defaults to `src/resources/retarget/calibration.npz`, with `calibration_qpos_deg`, `calibration_scale`,
  `calibration_rotation`, `metadata_json` containing `reference_frame`.
- `--reference`: defaults to `src/resources/retarget/reference.npy`, a matching finite `[T,21,3]` NPY in metres. Little-finger landmarks
  are synthetic because EX16 does not sense that finger.

The model, example calibration and reference data are bundled in both source and wheel distributions.
The calibration's numeric arrays are unchanged; legacy format naming and machine-specific
paths in `metadata_json` were replaced with a neutral format name and relative asset paths.
This repository has no recording, calibration capture, replay or training commands.
Array loading disables pickle;
weights must match the retained IK architecture and come from an authorized,
trusted source. Teleoperation defaults to CUDA; the offline slider viewer also
supports CPU inference. Collision checking requires
SAPIEN 2.x and the separate packaged GX16 collision URDF.

## Viewer Base Poses

`viewer.py` and `retarget_viewer.py` share a versioned JSON layout selected by
`--base-poses` (default `~/.local/share/gex16/base_poses.json`). Saving is explicit
and atomic; reset does not overwrite the saved file. A malformed file reports an
error and starts with defaults until explicitly saved or reloaded.

```json
{"version": 1, "bases": {
  "EX16": {"position": [-0.23, 0, 0], "wxyz": [1, 0, 0, 0]},
  "GX16": {"position": [0.23, 0, 0], "wxyz": [1, 0, 0, 0]}
}}
```

Positions are metres, quaternion order is w/x/y/z. Quaternions are normalized
on loading. Base poses transform scene roots only; they do not enter the model's
wrist-local projection or change physical robot coordinates. The offline viewer
has no serial or ZMQ hardware connection and emits no control messages.

## Initial EX16 Joints

The offline viewer saves its startup joint angles separately with **Save initial joints**.
`--initial-pose` selects the file (default `~/.local/share/gex16/ex16_initial_pose.json`).
It contains `version: 1`, `joint_names: ["joint1", ..., "joint16"]`, and `urdf_deg`
with 16 finite angles within the EX16 slider limits. Loading validates joint order
and limits; a bad file cannot change the current pose during reload. Startup uses
calibration if no valid saved pose is available. Saves are atomic and explicit;
zeroing sliders or loading calibration does not overwrite the saved pose.
