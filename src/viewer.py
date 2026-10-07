"""Viser display of live EX16 and GX16 joint states."""

import argparse
import json
import math
import os
import signal
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from paths import EX16_URDF_PATH, GX16_URDF_PATH
from transport import (
    EX16_ENDPOINT,
    EX16_TOPIC,
    GX16_ENDPOINT,
    GX16_TOPIC,
    JOINT_NAMES,
    decode_state,
    joint_values,
    subscribe,
)

DEFAULT_BASE_POSES = Path.home() / ".local/share/gex16/base_poses.json"


def default_base_poses():
    return {
        name: {"position": [x, 0.0, 0.0], "wxyz": [1.0, 0.0, 0.0, 0.0]}
        for name, x in (("EX16", -0.23), ("GX16", 0.23))
    }


def validate_base_poses(poses):
    if not isinstance(poses, dict) or set(poses) != {"EX16", "GX16"}:
        raise ValueError("Base poses must contain EX16 and GX16")
    normalized = {}
    for name, pose in poses.items():
        if not isinstance(pose, dict):
            raise ValueError(f"{name} pose must be an object")
        position = np.asarray(pose.get("position"), dtype=float)
        wxyz = np.asarray(pose.get("wxyz"), dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError(f"{name} position must contain three finite values")
        if wxyz.shape != (4,) or not np.isfinite(wxyz).all():
            raise ValueError(f"{name} quaternion must contain four finite values")
        norm = float(np.linalg.norm(wxyz))
        if not math.isfinite(norm) or norm < 1e-12:
            raise ValueError(f"{name} quaternion must be nonzero")
        normalized[name] = {
            "position": position.tolist(),
            "wxyz": (wxyz / norm).tolist(),
        }
    return normalized


def load_base_poses(path):
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("Unsupported base pose file version")
    return validate_base_poses(payload.get("bases"))


def save_base_poses(path, poses):
    save_json(path, {"version": 1, "bases": validate_base_poses(poses)})


def save_json(path, payload):
    """Replace a viewer settings file only after the new JSON is fully written."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as file:
            temporary = Path(file.name)
            json.dump(payload, file, indent=2, allow_nan=False)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class BasePoseControls:
    """Scene-only base transforms shared by the hardware and retarget viewers."""

    def __init__(self, server, path=DEFAULT_BASE_POSES):
        import viser
        import viser.transforms as tf

        self.path = Path(path).expanduser()
        self.controls = {}
        self._lock = threading.RLock()
        self._syncing = False
        poses = default_base_poses()
        message = "Default base poses"
        if self.path.exists():
            try:
                poses = load_base_poses(self.path)
                message = f"Loaded: {self.path}"
            except (OSError, ValueError, TypeError) as exc:
                message = f"Load failed: {exc}"
        with server.gui.add_folder("Base poses", expand_by_default=False):
            show = server.gui.add_checkbox("Gizmos", initial_value=True)
            save = server.gui.add_button("Save bases", icon=viser.Icon.DEVICE_FLOPPY)
            reload = server.gui.add_button("Reload bases", icon=viser.Icon.REFRESH)
            reset = server.gui.add_button("Reset bases", icon=viser.Icon.ARROW_BACK_UP)
            self.status = server.gui.add_markdown(message)
            for name, pose in poses.items():
                # JSON uses lists; older Viser releases require tuples or arrays.
                scene_pose = {key: tuple(value) for key, value in pose.items()}
                root = server.scene.add_frame(f"/{name}", show_axes=False, **scene_pose)
                gizmo = server.scene.add_transform_controls(
                    f"/gizmos/{name}",
                    scale=0.10,
                    depth_test=False,
                    **scene_pose,
                )
                with server.gui.add_folder(f"{name} base", expand_by_default=False):
                    position = server.gui.add_vector3(
                        "Position (m)",
                        initial_value=tuple(pose["position"]),
                        step=0.001,
                    )
                    rpy = server.gui.add_vector3(
                        "RPY (deg)",
                        initial_value=tuple(
                            np.rad2deg(tf.SO3(np.array(pose["wxyz"])).as_rpy_radians())
                        ),
                        step=1.0,
                    )
                self.controls[name] = (root, gizmo, position, rpy)

                @gizmo.on_update
                def move_base(event, name=name):
                    self.apply(name, event.target.position, event.target.wxyz)

                def edit_pose(event, name=name):
                    with self._lock:
                        if self._syncing:
                            return
                        _, _, position, rpy = self.controls[name]
                        rotation = tf.SO3.from_rpy_radians(*np.deg2rad(rpy.value))
                        self.apply(name, position.value, rotation.wxyz)

                position.on_update(edit_pose)
                rpy.on_update(edit_pose)

        @show.on_update
        def toggle_gizmos(event):
            for _, gizmo, _, _ in self.controls.values():
                gizmo.visible = event.target.value

        @save.on_click
        def save_clicked(_):
            try:
                save_base_poses(self.path, self.snapshot())
                self.status.content = f"Saved: {self.path}"
            except (OSError, ValueError, TypeError) as exc:
                self.status.content = f"Save failed: {exc}"

        @reload.on_click
        def reload_clicked(_):
            try:
                loaded = load_base_poses(self.path)
                for name, pose in loaded.items():
                    self.apply(name, **pose)
                self.status.content = f"Loaded: {self.path}"
            except (OSError, ValueError, TypeError) as exc:
                self.status.content = f"Load failed: {exc}"

        @reset.on_click
        def reset_clicked(_):
            for name, pose in default_base_poses().items():
                self.apply(name, **pose)
            self.status.content = "Default base poses (not saved)"

    def snapshot(self):
        with self._lock:
            return {
                name: {"position": root.position.tolist(), "wxyz": root.wxyz.tolist()}
                for name, (root, _, _, _) in self.controls.items()
            }

    def apply(self, name, position, wxyz):
        import viser.transforms as tf

        with self._lock:
            if self._syncing:
                return
            poses = self.snapshot()
            poses[name] = {"position": position, "wxyz": wxyz}
            pose = validate_base_poses(poses)[name]
            root, gizmo, xyz_input, rpy_input = self.controls[name]
            self._syncing = True
            try:
                root.position = gizmo.position = tuple(pose["position"])
                root.wxyz = gizmo.wxyz = tuple(pose["wxyz"])
                xyz_input.value = tuple(pose["position"])
                rpy_input.value = tuple(
                    np.rad2deg(tf.SO3(np.array(pose["wxyz"])).as_rpy_radians())
                )
            finally:
                self._syncing = False
            self.status.content = "Base poses changed (not saved)"


def load_urdf(path):
    import yourdfpy

    return yourdfpy.URDF.load(
        str(path),
        load_meshes=True,
        load_collision_meshes=False,
        filename_handler=lambda fname: str(path.parent / fname),
    )


def configuration(urdf, radians):
    angles = dict(zip(JOINT_NAMES, joint_values(radians)))
    if set(urdf.actuated_joint_names) != set(JOINT_NAMES):
        raise ValueError("URDF must expose joint1..joint16")
    return np.array([angles[name] for name in urdf.actuated_joint_names])


def configure_camera(server):
    server.scene.set_up_direction("+z")

    @server.on_client_connect
    def set_camera(client):
        aspect = None

        @client.camera.on_update
        def fit_width(camera):
            nonlocal aspect
            if aspect is not None and abs(camera.aspect - aspect) < 0.01:
                return
            aspect = camera.aspect
            target = np.array([0, 0, 0.08])
            if aspect < 0.8:
                # Leave the upper scene clear of the mobile controls.
                target[2] -= 0.55
            if hasattr(client.gui, "main_panel"):
                if aspect < 0.8:
                    client.gui.main_panel.float(
                        x=-8,
                        y=-8,
                        width=camera.image_width - 16,
                        height=camera.image_height * 0.42,
                    )
                elif camera.image_height:
                    client.gui.main_panel.float(x=-15, y=15, width=320, height=520)
            camera.position = target + np.array([0.66, -0.78, 0.47]) * max(
                1, 1 / aspect
            )
            camera.look_at = target
            camera.up_direction = (0, 0, 1)
            camera.fov = np.deg2rad(48)

        fit_width(client.camera)


def run(args, stop=None):
    import viser
    from viser.extras import ViserUrdf

    stop = stop or threading.Event()
    server = viser.ViserServer(host=args.host, port=args.port_viser)
    sockets = []
    try:
        configure_camera(server)
        BasePoseControls(server, args.base_poses)
        streams = []
        for name, path, endpoint, topic in (
            ("EX16", EX16_URDF_PATH, args.ex16_endpoint, EX16_TOPIC),
            ("GX16", GX16_URDF_PATH, args.gx16_endpoint, GX16_TOPIC),
        ):
            urdf = load_urdf(path)
            model = ViserUrdf(server, urdf, root_node_name=f"/{name}/model")
            model.update_cfg(configuration(urdf, np.zeros(16)))
            status = server.gui.add_markdown(f"**{name}:** waiting")
            socket = subscribe(endpoint, topic)
            sockets.append(socket)
            streams.append(
                dict(
                    socket=socket,
                    topic=topic,
                    urdf=urdf,
                    model=model,
                    status=status,
                    name=name,
                    received=None,
                )
            )
        while not stop.wait(0.01):
            for stream in streams:
                socket = stream["socket"]
                if socket.poll(0):
                    try:
                        state = decode_state(socket.recv_string(), stream["topic"])
                        stream["model"].update_cfg(
                            configuration(stream["urdf"], np.deg2rad(state["urdf_deg"]))
                        )
                        stream["received"] = time.monotonic()
                        stream["status"].content = f"**{stream['name']}:** live"
                    except (ValueError, TypeError) as exc:
                        stream[
                            "status"
                        ].content = f"**{stream['name']}:** invalid state ({exc})"
                received = stream["received"]
                if received is not None and time.monotonic() - received > args.timeout:
                    stream["status"].content = f"**{stream['name']}:** disconnected"
    finally:
        for socket in sockets:
            socket.close()
        server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ex16-endpoint", default=EX16_ENDPOINT)
    parser.add_argument("--gx16-endpoint", default=GX16_ENDPOINT)
    parser.add_argument("--timeout", type=float, default=3)
    parser.add_argument("--base-poses", type=Path, default=DEFAULT_BASE_POSES)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port-viser", type=int, default=8080)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    run(args, stop)


if __name__ == "__main__":
    main()
