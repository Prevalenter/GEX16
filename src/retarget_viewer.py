"""Offline EX16 joint sliders and GEX Retarget visualization. No hardware I/O."""

import argparse
import json
import math
import signal
import threading
import time
from pathlib import Path

import numpy as np

from paths import (
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_REFERENCE_PATH,
    EX16_URDF_PATH,
    GX16_URDF_PATH,
)
from retargeting.model import load_model
from retargeting.projection import restore_projector
from transport import JOINT_NAMES, joint_values
from viewer import (
    DEFAULT_BASE_POSES,
    BasePoseControls,
    configuration,
    configure_camera,
    load_urdf,
    save_json,
)

DEFAULT_INITIAL_POSE = Path.home() / ".local/share/gex16/ex16_initial_pose.json"


def validate_initial_pose(payload, lower, upper):
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("Unsupported initial pose file version")
    if payload.get("joint_names") != list(JOINT_NAMES):
        raise ValueError("Initial pose must use joint1..joint16 order")
    angles = joint_values(payload.get("urdf_deg"))
    if np.any(angles < lower) or np.any(angles > upper):
        raise ValueError("Initial pose exceeds EX16 joint limits")
    return angles


def load_initial_pose(path, lower, upper):
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return validate_initial_pose(payload, lower, upper)


def save_initial_pose(path, angles, lower, upper):
    payload = {
        "version": 1,
        "joint_names": list(JOINT_NAMES),
        "urdf_deg": joint_values(angles).tolist(),
    }
    validate_initial_pose(payload, lower, upper)
    save_json(path, payload)


def slider_limits(urdf):
    limits = np.array(
        [
            [urdf.joint_map[name].limit.lower, urdf.joint_map[name].limit.upper]
            for name in JOINT_NAMES
        ],
        dtype=float,
    )
    if not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("EX16 requires finite, ordered joint limits")
    degrees = np.rad2deg(limits)
    return np.ceil(degrees[:, 0] * 10) / 10, np.floor(degrees[:, 1] * 10) / 10


def retarget_pose(projector, model, ex16_degrees):
    human = projector.project(joint_values(ex16_degrees))
    qpos = joint_values(model.forward(human))
    return np.clip(
        qpos,
        model.qpos_normalizer.joint_lower_limit,
        model.qpos_normalizer.joint_upper_limit,
    )


def run(args, stop=None):
    import viser
    from viser.extras import ViserUrdf

    stop = stop or threading.Event()
    projector, _ = restore_projector(args.calibration, args.reference)
    model = load_model(args.checkpoint_dir, device=args.device)
    ex16, gx16 = load_urdf(EX16_URDF_PATH), load_urdf(GX16_URDF_PATH)
    lower, upper = slider_limits(ex16)
    calibrated = np.clip(np.round(projector.calibration_qpos_deg, 1), lower, upper)
    initial_path = Path(args.initial_pose).expanduser()
    initial = calibrated
    initial_status = "Initial joints: calibration pose"
    if initial_path.exists():
        try:
            initial = load_initial_pose(initial_path, lower, upper)
            initial_status = f"Loaded initial joints: {initial_path}"
        except (OSError, ValueError, TypeError) as exc:
            initial_status = f"Initial pose load failed: {exc}; using calibration pose"
    server = viser.ViserServer(host=args.host, port=args.port_viser)
    try:
        configure_camera(server)
        BasePoseControls(server, args.base_poses)
        source = ViserUrdf(server, ex16, root_node_name="/EX16/model")
        target = ViserUrdf(server, gx16, root_node_name="/GX16/model")
        status = server.gui.add_markdown(f"**GEX Retarget:** {model.device}")
        with server.gui.add_folder("EX16 joints"):
            zero = server.gui.add_button("Zero joints", icon=viser.Icon.ARROW_BACK_UP)
            calibration = server.gui.add_button(
                "Calibration pose", icon=viser.Icon.REFRESH
            )
            save = server.gui.add_button(
                "Save initial joints", icon=viser.Icon.DEVICE_FLOPPY
            )
            reload = server.gui.add_button(
                "Reload initial joints", icon=viser.Icon.REFRESH
            )
            pose_status = server.gui.add_markdown(initial_status)
            sliders = []
            for finger, name in enumerate(("Thumb", "Index", "Middle", "Ring")):
                with server.gui.add_folder(name, expand_by_default=finger == 0):
                    for index in range(finger * 4, finger * 4 + 4):
                        sliders.append(
                            server.gui.add_slider(
                                f"J{index + 1:02d} (deg)",
                                min=float(lower[index]),
                                max=float(upper[index]),
                                step=0.1,
                                initial_value=float(initial[index]),
                            )
                        )
        with server.gui.add_folder("GX16 output (deg)", expand_by_default=False):
            output = server.gui.add_markdown("")
        lock = threading.Lock()

        def set_joints(values):
            with lock:
                for slider, value in zip(sliders, np.clip(values, lower, upper)):
                    slider.value = float(value)

        zero.on_click(lambda _: set_joints(np.zeros(16)))
        calibration.on_click(lambda _: set_joints(calibrated))

        @save.on_click
        def save_clicked(_):
            try:
                with lock:
                    values = [slider.value for slider in sliders]
                save_initial_pose(initial_path, values, lower, upper)
                pose_status.content = f"Saved initial joints: {initial_path}"
            except (OSError, ValueError, TypeError) as exc:
                pose_status.content = f"Initial pose save failed: {exc}"

        @reload.on_click
        def reload_clicked(_):
            try:
                set_joints(load_initial_pose(initial_path, lower, upper))
                pose_status.content = f"Loaded initial joints: {initial_path}"
            except (OSError, ValueError, TypeError) as exc:
                pose_status.content = f"Initial pose load failed: {exc}"

        previous = None
        while not stop.is_set():
            with lock:
                angles = np.array([slider.value for slider in sliders])
            if previous is None or not np.array_equal(angles, previous):
                source.update_cfg(configuration(ex16, np.deg2rad(angles)))
                started = time.perf_counter()
                try:
                    qpos = retarget_pose(projector, model, angles)
                    target.update_cfg(configuration(gx16, qpos))
                    degrees = np.rad2deg(qpos).reshape(4, 4)
                    output.content = "\n\n".join(
                        f"**{finger}:** "
                        + ", ".join(f"{value:.1f}" for value in values)
                        for finger, values in zip(
                            ("Thumb", "Index", "Middle", "Ring"), degrees
                        )
                    )
                    status.content = f"**GEX Retarget:** {model.device} | {(time.perf_counter() - started) * 1000:.1f} ms"
                except (ValueError, RuntimeError) as exc:
                    status.content = f"**Retarget failed:** {exc} (GX16 unchanged)"
                previous = angles.copy()
            stop.wait(1 / args.update_hz)
    finally:
        server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Model directory (default: bundled GEX Retarget model).",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION_PATH,
        help="Calibration NPZ (default: bundled example).",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE_PATH,
        help="Reference NPY (default: bundled example).",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--base-poses", type=Path, default=DEFAULT_BASE_POSES)
    parser.add_argument("--initial-pose", type=Path, default=DEFAULT_INITIAL_POSE)
    parser.add_argument("--update-hz", type=float, default=20)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port-viser", type=int, default=8081)
    args = parser.parse_args()
    if not math.isfinite(args.update_hz) or not 0 < args.update_hz <= 120:
        parser.error("--update-hz must be finite and in (0, 120]")
    if not 0 <= args.port_viser <= 65535:
        parser.error("--port-viser must be in 0..65535")
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    run(args, stop)


if __name__ == "__main__":
    main()
