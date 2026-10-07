"""Hardware paths are exercised only with injected test objects."""

import argparse
import json
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import zmq

import device
from ex16 import Glove16
from ex16 import run as run_ex16
from gx16 import JOINT_DIRECTIONS, Hand16, handle_request
from gx16 import run as run_gx16
from motor import Motor
from transport import GX16Client, decode_state, joint_values


@pytest.mark.parametrize(
    "module", ["ex16", "gx16", "viewer", "teleop", "retarget_viewer"]
)
@pytest.mark.parametrize("mode", ["module", "script"])
def test_help(module, mode, tmp_path):
    target = (
        ["-m", module]
        if mode == "module"
        else [str(Path(__file__).resolve().parents[1] / "src" / f"{module}.py")]
    )
    result = subprocess.run(
        [sys.executable, *target, "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize("driver", [Glove16, Hand16])
def test_explicit_device_required(driver):
    with pytest.raises(ValueError):
        driver()


@pytest.mark.parametrize("values", [[0] * 15, [float("nan")] * 16, [float("inf")] * 16])
def test_bad_joints(values):
    with pytest.raises(ValueError):
        joint_values(values)


def test_driver_angles_and_cleanup(monkeypatch):
    handler = Mock()
    motors = [Mock(get_pos=Mock(return_value=100)) for _ in range(16)]
    monkeypatch.setattr(device, "connect", lambda *a: (handler, motors))
    glove = Glove16(port="test-only")
    glove.connect()
    np.testing.assert_allclose(glove.getjs(), 10 * glove.directions)
    glove.close()
    assert not glove.is_connected
    handler.closePort.assert_called_once()
    hand = Hand16(port="test-only")
    hand.connect()
    hand.setjs(range(16))
    for i, motor in enumerate(motors):
        motor.set_pos.assert_called_with(i + 90)
    with pytest.raises(ValueError):
        hand.setjs([0])
    with pytest.raises(ValueError):
        hand.setj(17, 0)
    hand.close()


def test_partial_initialization_cleanup(monkeypatch):
    handler = Mock()
    motors = [Mock() for _ in range(16)]
    motors[1].init_config.side_effect = RuntimeError("initialization failed")
    monkeypatch.setattr(device, "connect", lambda *a: (handler, motors))
    hand = Hand16(port="test-only")
    with pytest.raises(RuntimeError):
        hand.connect()
    assert all(m.torq_off.called for m in motors)
    handler.closePort.assert_called_once()
    assert not hand.is_connected


def test_cleanup_attempts_every_motor():
    handler, first, last = Mock(), Mock(), Mock()
    first.torq_off.side_effect = RuntimeError("write failure")
    with pytest.raises(RuntimeError):
        device.close(handler, [first, last])
    last.torq_off.assert_called_once()
    handler.closePort.assert_called_once()


def test_serial_setup_failure_closes_port(monkeypatch):
    handler = Mock()
    handler.openPort.return_value = True
    handler.setBaudRate.return_value = False
    monkeypatch.setattr(device, "PortHandler", lambda port: handler)
    with pytest.raises(RuntimeError):
        device.connect("test-only", device.config("ex16"))
    handler.closePort.assert_called_once()


def test_motor_register_values(monkeypatch):
    monkeypatch.setattr("motor.time.sleep", lambda _: None)
    packet = Mock()
    packet.read4ByteTxRx.return_value = (2**32 - 1, 0, 0)
    motor = Motor(1, "port", packet)
    assert motor.get_pos() == pytest.approx(-0.087891)
    motor.init_config(800, 500, 300)
    packet.write2ByteTxRx.assert_any_call("port", 1, 38, 800)
    packet.write2ByteTxRx.assert_any_call("port", 1, 102, 500)
    packet.write2ByteTxRx.assert_any_call("port", 1, 100, 300)
    packet.write1ByteTxRx.assert_any_call("port", 1, 11, 5)


def test_commands_validate_before_write():
    hand, stop = Mock(), threading.Event()
    request = {"cmd": "setjs", "positions": list(range(16))}
    assert handle_request(hand, request, stop)["ok"]
    np.testing.assert_allclose(
        hand.setjs.call_args.args[0], np.arange(16) * JOINT_DIRECTIONS
    )
    for request in (
        {"cmd": "setj", "joint": 1.5, "position": 0},
        {"cmd": "setjs", "positions": [0]},
        {"cmd": "missing"},
    ):
        assert not handle_request(hand, request, stop)["ok"]
    assert handle_request(hand, {"cmd": "shutdown"}, stop)["ok"]
    assert stop.is_set()


def endpoint():
    socket = zmq.Context.instance().socket(zmq.REP)
    port = socket.bind_to_random_port("tcp://127.0.0.1")
    socket.close(0)
    return f"tcp://127.0.0.1:{port}"


def test_real_gx16_loop_with_injected_hand():
    stop = threading.Event()
    hand = Mock(port="test-only", is_connected=True)
    hand.getjs.return_value = [0] * 16
    args = argparse.Namespace(
        cmd_endpoint=endpoint(),
        state_endpoint=endpoint(),
        curr_limit=800,
        goal_current=500,
        goal_pwm=300,
        state_hz=20,
    )
    failures = []

    def worker():
        try:
            run_gx16(args, hand, stop)
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    client = GX16Client(args.cmd_endpoint, 2000)
    try:
        assert client.request("ping")["connected"]
        np.testing.assert_allclose(client.get_qpos(), 0)
        client.set_qpos(np.zeros(16))
        client.request("shutdown")
    finally:
        stop.set()
        client.close()
        thread.join(5)
    assert not thread.is_alive()
    assert not failures
    hand.close.assert_called_once()


def test_ex16_loop_with_injected_glove():
    stop = threading.Event()
    glove = Mock(port="test-only")
    glove.getjs.return_value = [0] * 16
    args = argparse.Namespace(state_endpoint=endpoint(), state_hz=100)
    subscriber = zmq.Context.instance().socket(zmq.SUB)
    subscriber.linger = 0
    subscriber.setsockopt_string(zmq.SUBSCRIBE, "ex16/state")
    subscriber.connect(args.state_endpoint)
    failures = []

    def worker():
        try:
            run_ex16(args, glove, stop)
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert subscriber.poll(3000)
        state = decode_state(subscriber.recv_string(), "ex16/state")
        np.testing.assert_allclose(state["urdf_deg"], 0)
    finally:
        stop.set()
        subscriber.close()
        thread.join(5)
    assert not thread.is_alive()
    assert not failures
    glove.close.assert_called_once()


def test_decode_rejects_invalid_messages():
    with pytest.raises(ValueError):
        decode_state("wrong {}", "ex16/state")
    with pytest.raises(ValueError):
        decode_state(
            "ex16/state "
            + json.dumps({"timestamp": float("nan"), "urdf_deg": [0] * 16}),
            "ex16/state",
        )
