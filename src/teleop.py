"""EX16 -> GEX Retarget -> GX16, with Viser preview and optional hardware output."""

import argparse
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
    GX16_URDF_PATH,
)
from retargeting.control import smooth_gx16_command
from retargeting.model import get_config, load_model
from retargeting.projection import TIP_IDS, restore_projector
from transport import (
    COMMAND_ENDPOINT,
    EX16_ENDPOINT,
    EX16_TOPIC,
    GX16Client,
    decode_state,
    joint_values,
    subscribe,
)
from viewer import configuration, configure_camera, load_urdf


def run(args, stop=None):
    import viser
    from viser.extras import ViserUrdf

    stop = stop or threading.Event()
    projector, _ = restore_projector(args.calibration, args.reference)
    model = load_model(args.checkpoint_dir)
    lower = np.asarray(model.qpos_normalizer.joint_lower_limit)
    upper = np.asarray(model.qpos_normalizer.joint_upper_limit)
    collision_model = None
    if not args.no_collision_check:
        from _vendor.geort.env.hand import HandKinematicModel

        collision_model = HandKinematicModel.build_from_config(get_config())

    server = client = subscriber = None
    try:
        previous = None
        if args.enable_gx16_output:
            client = GX16Client(
                args.gx16_command_endpoint, args.gx16_command_timeout_ms
            )
            previous = client.get_qpos()
        server = viser.ViserServer(host=args.host, port=args.port_viser)
        configure_camera(server)
        urdf = load_urdf(GX16_URDF_PATH)
        robot = ViserUrdf(server, urdf, root_node_name="/gx16")
        targets = server.scene.add_point_cloud(
            "/targets",
            points=np.zeros((4, 3), dtype=np.float32),
            colors=np.array(
                [[245, 158, 11], [34, 197, 94], [59, 130, 246], [239, 68, 68]],
                dtype=np.uint8,
            ),
            point_size=0.006,
        )
        enabled = server.gui.add_checkbox("Live updates", initial_value=True)
        collision_filter = server.gui.add_checkbox(
            "Block colliding commands", initial_value=False
        )
        collision_filter.disabled = collision_model is None
        status = server.gui.add_markdown("Waiting for EX16")
        subscriber = subscribe(args.ex16_endpoint, EX16_TOPIC)
        latest, received, next_update = None, time.monotonic(), 0
        while not stop.is_set():
            if subscriber.poll(10):
                try:
                    latest = decode_state(subscriber.recv_string(), EX16_TOPIC)
                    received = time.monotonic()
                except (TypeError, ValueError) as exc:
                    status.content = f"Invalid EX16 state: {exc}"
            now = time.monotonic()
            if now - received > args.timeout:
                status.content = "EX16 disconnected; no new commands"
                continue
            if latest is None or not enabled.value or now < next_update:
                continue
            next_update = now + 1 / args.update_hz
            human = projector.project(latest["urdf_deg"])
            desired = np.clip(joint_values(model.forward(human)), lower, upper)
            previous = (
                desired
                if previous is None
                else smooth_gx16_command(
                    desired,
                    previous,
                    lower,
                    upper,
                    args.smoothing_alpha,
                )
            )
            collision = (
                None
                if collision_model is None
                else collision_model.has_self_collision(
                    previous,
                    penetration_threshold=args.collision_threshold_mm / 1000,
                )
            )
            result = "Preview only"
            if client is not None:
                if collision and collision_filter.value:
                    result = "Blocked: self-collision"
                else:
                    # Communication errors terminate the loop and trigger torque-off.
                    client.set_qpos(previous)
                    result = "Commanding GX16"
            robot.update_cfg(configuration(urdf, previous))
            targets.points = human[TIP_IDS].astype(np.float32)
            status.content = f"{result} | Collision: {collision if collision is not None else 'not checked'}"
    finally:
        try:
            if client is not None:
                try:
                    client.request("torque_off")
                finally:
                    client.close()
        finally:
            if subscriber is not None:
                subscriber.close()
            if server is not None:
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
    parser.add_argument("--ex16-endpoint", default=EX16_ENDPOINT)
    parser.add_argument("--gx16-command-endpoint", default=COMMAND_ENDPOINT)
    parser.add_argument("--gx16-command-timeout-ms", type=int, default=500)
    parser.add_argument("--enable-gx16-output", action="store_true")
    parser.add_argument("--update-hz", type=float, default=10)
    parser.add_argument("--timeout", type=float, default=5)
    parser.add_argument("--smoothing-alpha", type=float, default=0.35)
    parser.add_argument("--collision-threshold-mm", type=float, default=0.5)
    parser.add_argument("--no-collision-check", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port-viser", type=int, default=8080)
    args = parser.parse_args()
    for name in ("update_hz", "timeout", "smoothing_alpha"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.smoothing_alpha > 1:
        parser.error("--smoothing-alpha must be at most 1")
    if (
        not math.isfinite(args.collision_threshold_mm)
        or args.collision_threshold_mm < 0
    ):
        parser.error("--collision-threshold-mm must be finite and non-negative")
    if args.gx16_command_timeout_ms <= 0:
        parser.error("--gx16-command-timeout-ms must be positive")
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    run(args, stop)


if __name__ == "__main__":
    main()
