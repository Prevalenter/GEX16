import argparse
import importlib
import json
import threading
import xml.etree.ElementTree as ET
from unittest.mock import Mock

import numpy as np
import pytest

from paths import (
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_REFERENCE_PATH,
    EX16_URDF_PATH,
    GX16_URDF_PATH,
    RESOURCE_ROOT,
)
from retargeting.control import smooth_gx16_command
from retargeting.model import load_model


def test_resources():
    paths = list(RESOURCE_ROOT.rglob("*.urdf"))
    assert len(paths) == 3
    for path in paths:
        for mesh in ET.parse(path).findall(".//mesh"):
            assert (path.parent / mesh.attrib["filename"]).is_file()


def test_smoothing_and_limits():
    result = smooth_gx16_command([4] * 16, [0] * 16, [-1] * 16, [1] * 16, 0.25)
    np.testing.assert_allclose(result, 0.25)
    with pytest.raises(ValueError):
        smooth_gx16_command([0] * 16, [0] * 16, [-1] * 16, [1] * 16, 0)


def test_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path)


def test_bundled_checkpoint_files():
    assert (DEFAULT_CHECKPOINT_DIR / "last.pth").is_file()
    config = json.loads((DEFAULT_CHECKPOINT_DIR / "config.json").read_text())
    assert config["joint_order"] == [f"joint{i}" for i in range(1, 17)]
    assert len(config["joint"]["lower"]) == len(config["joint"]["upper"]) == 16
    assert "robot_data" not in config
    assert (DEFAULT_CHECKPOINT_DIR / config["urdf_path"]).is_file()


def test_default_model_inference():
    pytest.importorskip("torch")
    model = load_model(device="cpu")
    qpos = model.forward(np.zeros((21, 3), dtype=np.float32))
    assert qpos.shape == (16,) and np.isfinite(qpos).all()
    assert np.all(qpos >= model.qpos_normalizer.joint_lower_limit)
    assert np.all(qpos <= model.qpos_normalizer.joint_upper_limit)


def test_bundled_calibration_and_reference():
    reference = np.load(DEFAULT_REFERENCE_PATH, allow_pickle=False)
    assert reference.shape[1:] == (21, 3) and np.isfinite(reference).all()
    with np.load(DEFAULT_CALIBRATION_PATH, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        assert metadata["format"] == "gex16.ex16_calibration"
        assert 0 <= metadata["reference_frame"] < len(reference)
        assert (
            DEFAULT_CALIBRATION_PATH.parent / metadata["reference_path"]
        ).resolve() == DEFAULT_REFERENCE_PATH.resolve()
        assert (
            DEFAULT_CALIBRATION_PATH.parent / metadata["urdf_path"]
        ).resolve() == EX16_URDF_PATH.resolve()
        assert archive["calibration_qpos_deg"].shape == (16,)
        assert archive["calibration_rotation"].shape == (3, 3)
        assert archive["qpos_deg"].shape[1] == 16
        assert "/home/" not in str(metadata) and "/Users/" not in str(metadata)


def test_bundled_projection_and_inference():
    pytest.importorskip("torch")
    pytest.importorskip("yourdfpy")
    from retargeting.projection import restore_projector

    projector, _ = restore_projector(DEFAULT_CALIBRATION_PATH, DEFAULT_REFERENCE_PATH)
    model = load_model(device="cpu")
    output = model.forward(projector.project(projector.calibration_qpos_deg))
    assert output.shape == (16,) and np.isfinite(output).all()


@pytest.mark.parametrize("name", ["teleop", "retarget_viewer"])
def test_cli_defaults_use_bundled_inputs(monkeypatch, name):
    module = importlib.import_module(name)
    captured = Mock()
    monkeypatch.setattr(module, "run", captured)
    monkeypatch.setattr(module.signal, "signal", lambda *_: None)
    monkeypatch.setattr("sys.argv", [name])
    module.main()
    args = captured.call_args.args[0]
    assert args.checkpoint_dir == DEFAULT_CHECKPOINT_DIR
    assert args.calibration == DEFAULT_CALIBRATION_PATH
    assert args.reference == DEFAULT_REFERENCE_PATH


def test_projection_restore_and_models(tmp_path):
    pytest.importorskip("yourdfpy")
    from retargeting.projection import EX16HumanProjector, restore_projector
    from viewer import configuration, load_urdf

    for path in (EX16_URDF_PATH, GX16_URDF_PATH):
        urdf = load_urdf(path)
        assert len(urdf.scene.geometry) > 0
        np.testing.assert_allclose(configuration(urdf, np.zeros(16)), 0)
    projector = EX16HumanProjector()
    reference = projector.source_keypoints(np.zeros(16))[None]
    np.save(tmp_path / "reference.npy", reference)
    np.savez(
        tmp_path / "calibration.npz",
        calibration_qpos_deg=np.zeros(16),
        calibration_scale=1.0,
        calibration_rotation=np.eye(3),
        metadata_json=json.dumps({"reference_frame": 0}),
    )
    restored, _ = restore_projector(
        tmp_path / "calibration.npz", tmp_path / "reference.npy"
    )
    np.testing.assert_allclose(restored.project(np.zeros(16)), reference[0], atol=1e-8)


def test_teleop_command_failure_disables_torque(monkeypatch):
    viser = pytest.importorskip("viser")
    from viser import extras

    import teleop

    args = argparse.Namespace(
        calibration="unused",
        reference="unused",
        checkpoint_dir="unused",
        no_collision_check=True,
        enable_gx16_output=True,
        gx16_command_endpoint="unused",
        gx16_command_timeout_ms=500,
        host="127.0.0.1",
        port_viser=8080,
        ex16_endpoint="unused",
        timeout=5,
        update_hz=10,
        smoothing_alpha=0.35,
        collision_threshold_mm=0.5,
    )
    projector = Mock()
    projector.project.return_value = np.zeros((21, 3))
    monkeypatch.setattr(teleop, "restore_projector", lambda *a: (projector, {}))
    model = Mock()
    model.qpos_normalizer.joint_lower_limit = np.full(16, -1.0)
    model.qpos_normalizer.joint_upper_limit = np.ones(16)
    model.forward.return_value = np.zeros(16)
    monkeypatch.setattr(teleop, "load_model", lambda *a: model)
    client = Mock()
    client.get_qpos.return_value = np.zeros(16)
    client.set_qpos.side_effect = RuntimeError("command timeout")
    monkeypatch.setattr(teleop, "GX16Client", lambda *a: client)
    server = Mock()
    server.gui.add_checkbox.return_value.value = True
    monkeypatch.setattr(viser, "ViserServer", lambda **kw: server)
    monkeypatch.setattr(extras, "ViserUrdf", Mock())
    monkeypatch.setattr(teleop, "load_urdf", lambda *a: Mock())
    subscriber = Mock()
    subscriber.poll.return_value = True
    subscriber.recv_string.return_value = "ex16/state " + json.dumps(
        {"urdf_deg": [0] * 16, "timestamp": 1}
    )
    monkeypatch.setattr(teleop, "subscribe", lambda *a: subscriber)
    with pytest.raises(RuntimeError, match="command timeout"):
        teleop.run(args, threading.Event())
    client.request.assert_called_once_with("torque_off")
    client.close.assert_called_once()
    subscriber.close.assert_called_once()
    server.stop.assert_called_once()
