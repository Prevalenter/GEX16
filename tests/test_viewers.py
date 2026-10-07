import copy
import json
from unittest.mock import Mock

import numpy as np
import pytest

from viewer import (
    BasePoseControls,
    default_base_poses,
    load_base_poses,
    save_base_poses,
    validate_base_poses,
)


def test_pose_file_roundtrip(tmp_path):
    poses = default_base_poses()
    poses["EX16"] = {"position": [1, 2, 3], "wxyz": [2, 0, 0, 0]}
    path = tmp_path / "nested/bases.json"
    save_base_poses(path, poses)
    loaded = load_base_poses(path)
    assert loaded["EX16"] == {"position": [1, 2, 3], "wxyz": [1, 0, 0, 0]}
    assert loaded["GX16"] == poses["GX16"]
    assert default_base_poses()["EX16"]["position"] == [-0.23, 0, 0]


@pytest.mark.parametrize(
    "field,value",
    [
        ("position", [0, 1]),
        ("position", [0, 0, float("nan")]),
        ("wxyz", [0, 0, 0, 0]),
        ("wxyz", [float("inf"), 0, 0, 0]),
    ],
)
def test_bad_pose_does_not_overwrite(tmp_path, field, value):
    path = tmp_path / "bases.json"
    poses = default_base_poses()
    save_base_poses(path, poses)
    poses["EX16"][field] = value
    with pytest.raises(ValueError):
        save_base_poses(path, poses)
    assert load_base_poses(path) == default_base_poses()


def test_failed_replace_keeps_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "bases.json"
    save_base_poses(path, default_base_poses())
    import viewer

    monkeypatch.setattr(viewer.os, "replace", Mock(side_effect=OSError("denied")))
    with pytest.raises(OSError):
        save_base_poses(path, default_base_poses())
    assert load_base_poses(path) == default_base_poses()
    assert list(tmp_path.iterdir()) == [path]


def test_invalid_document():
    with pytest.raises(ValueError):
        validate_base_poses({})


@pytest.mark.parametrize("saved", [False, True])
def test_scene_poses_use_viser_compatible_types(tmp_path, monkeypatch, saved):
    viser = pytest.importorskip("viser")
    server = viser.ViserServer(host="127.0.0.1", port=0)
    path = tmp_path / "bases.json"
    poses = default_base_poses()
    if saved:
        poses["EX16"] = {"position": [0.1, 0.2, 0.3], "wxyz": [0, 0, 0, 2]}
        save_base_poses(path, poses)
    calls = []

    def strict_vectors(method):
        def wrapped(*args, **kwargs):
            # Viser may also create ancestor frames with its own default pose.
            if "position" in kwargs or "wxyz" in kwargs:
                assert isinstance(kwargs["position"], tuple)
                assert isinstance(kwargs["wxyz"], tuple)
                calls.append(args[0])
            return method(*args, **kwargs)

        return wrapped

    for name in ("add_frame", "add_transform_controls"):
        monkeypatch.setattr(
            server.scene, name, strict_vectors(getattr(server.scene, name))
        )
    try:
        controls = BasePoseControls(server, path)
        assert set(calls) == {"/EX16", "/GX16", "/gizmos/EX16", "/gizmos/GX16"}
        assert controls.snapshot() == validate_base_poses(poses)
    finally:
        server.stop()


def test_controls_share_root_pose_and_restore(tmp_path):
    viser = pytest.importorskip("viser")
    server = viser.ViserServer(host="127.0.0.1", port=0)
    try:
        path = tmp_path / "bases.json"
        controls = BasePoseControls(server, path)
        controls.apply("EX16", [0.5, 0, 0], [0, 0, 0, 1])
        root, gizmo, position, rpy = controls.controls["EX16"]
        np.testing.assert_allclose(root.position, gizmo.position)
        np.testing.assert_allclose(root.wxyz, gizmo.wxyz)
        position.value = (0.7, 0.0, 0.0)
        np.testing.assert_allclose(root.position, [0.7, 0, 0])
        rpy.value = (0.0, 0.0, 90.0)
        np.testing.assert_allclose(root.wxyz, [2**-0.5, 0, 0, 2**-0.5], atol=1e-8)
        saved = controls.snapshot()
        save_base_poses(path, saved)
        restored = BasePoseControls(server, path)
        for name in saved:
            np.testing.assert_allclose(
                restored.snapshot()[name]["position"], saved[name]["position"]
            )
            np.testing.assert_allclose(
                restored.snapshot()[name]["wxyz"], saved[name]["wxyz"]
            )
    finally:
        server.stop()


def test_slider_limits_and_retarget_values():
    pytest.importorskip("yourdfpy")
    from paths import EX16_URDF_PATH
    from retarget_viewer import retarget_pose, slider_limits
    from viewer import load_urdf

    lower, upper = slider_limits(load_urdf(EX16_URDF_PATH))
    assert lower.shape == upper.shape == (16,)
    assert np.all(lower < 0) and np.all(upper > 0)
    projector, model = Mock(), Mock()
    projector.project.return_value = np.zeros((21, 3))
    model.forward.return_value = np.linspace(-2, 2, 16)
    model.qpos_normalizer.joint_lower_limit = np.full(16, -1.0)
    model.qpos_normalizer.joint_upper_limit = np.ones(16)
    result = retarget_pose(projector, model, np.arange(16))
    np.testing.assert_array_equal(projector.project.call_args.args[0], np.arange(16))
    np.testing.assert_allclose(result, np.clip(model.forward.return_value, -1, 1))
    model.forward.return_value[0] = float("nan")
    with pytest.raises(ValueError):
        retarget_pose(projector, model, np.zeros(16))


def test_cpu_model_and_cuda_default(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from _vendor.geort.model import IKModel
    from _vendor.geort.utils.config_utils import parse_config_keypoint_info
    from retargeting.model import get_config, load_model

    config = copy.deepcopy(get_config())
    config["joint"] = {"lower": [-1] * 16, "upper": [1] * 16}
    (tmp_path / "config.json").write_text(json.dumps(config))
    info = parse_config_keypoint_info(config)
    torch.manual_seed(1)
    original = IKModel(info["joint"]).eval()
    torch.save(original.state_dict(), tmp_path / "last.pth")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = load_model(tmp_path, device="auto")
    assert model.device.type == "cpu"
    human = np.arange(63, dtype=np.float32).reshape(21, 3) / 100
    with torch.inference_mode():
        expected = original(torch.from_numpy(human[info["human_id"]])[None]).numpy()[0]
    np.testing.assert_allclose(model.forward(human), expected, atol=1e-7)
    with pytest.raises(RuntimeError, match="CUDA"):
        load_model(tmp_path)
    with pytest.raises(ValueError, match="device"):
        load_model(tmp_path, device="invalid")


def test_initial_joint_pose_roundtrip(tmp_path):
    from retarget_viewer import load_initial_pose, save_initial_pose

    lower, upper = np.full(16, -170.0), np.full(16, 170.0)
    angles = np.arange(16, dtype=float) - 7.5
    path = tmp_path / "settings/initial.json"
    save_initial_pose(path, angles, lower, upper)
    np.testing.assert_array_equal(load_initial_pose(path, lower, upper), angles)
    assert json.loads(path.read_text())["joint_names"] == [
        f"joint{i}" for i in range(1, 17)
    ]


@pytest.mark.parametrize(
    "angles",
    [[0] * 15, [float("nan")] * 16, [float("inf")] * 16, [200] * 16, [-200] * 16],
)
def test_bad_initial_pose_preserves_saved_file(tmp_path, angles):
    from retarget_viewer import load_initial_pose, save_initial_pose

    lower, upper = np.full(16, -170.0), np.full(16, 170.0)
    path = tmp_path / "initial.json"
    save_initial_pose(path, np.zeros(16), lower, upper)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        save_initial_pose(path, angles, lower, upper)
    assert path.read_bytes() == before
    np.testing.assert_array_equal(load_initial_pose(path, lower, upper), np.zeros(16))


def test_initial_pose_rejects_wrong_order_and_version(tmp_path):
    from retarget_viewer import load_initial_pose, save_initial_pose

    lower, upper = np.full(16, -170.0), np.full(16, 170.0)
    path = tmp_path / "initial.json"
    save_initial_pose(path, np.zeros(16), lower, upper)
    payload = json.loads(path.read_text())
    payload["joint_names"].reverse()
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="order"):
        load_initial_pose(path, lower, upper)
    payload["version"] = 2
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="version"):
        load_initial_pose(path, lower, upper)
