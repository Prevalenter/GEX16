"""Small ZeroMQ helpers. Wire angles are degrees; retargeting uses radians."""

import json
import math

import numpy as np
import zmq

EX16_ENDPOINT = "tcp://127.0.0.1:5567"
GX16_ENDPOINT = "tcp://127.0.0.1:5557"
COMMAND_ENDPOINT = "tcp://127.0.0.1:5556"
EX16_TOPIC = "ex16/state"
GX16_TOPIC = "gx16/state"
JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 17))


def joint_values(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (16,) or not np.isfinite(values).all():
        raise ValueError("Expected 16 finite joint values")
    return values


def publish(socket, topic, payload):
    socket.send_string(f"{topic} {json.dumps(payload, allow_nan=False)}")


def subscribe(endpoint, topic):
    socket = zmq.Context.instance().socket(zmq.SUB)
    socket.linger = 0
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt_string(zmq.SUBSCRIBE, topic)
    socket.connect(endpoint)
    return socket


def decode_state(message, topic):
    received, encoded = message.split(" ", 1)
    if received != topic:
        raise ValueError(f"Expected topic {topic!r}")
    payload = json.loads(encoded)
    if not isinstance(payload, dict):
        raise ValueError("State must be a JSON object")
    payload["urdf_deg"] = joint_values(payload.get("urdf_deg"))
    if not math.isfinite(float(payload.get("timestamp", math.nan))):
        raise ValueError("State timestamp must be finite")
    return payload


class GX16Client:
    def __init__(self, endpoint=COMMAND_ENDPOINT, timeout_ms=500):
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        self.endpoint, self.timeout_ms = endpoint, timeout_ms
        self.socket = None
        self._reset()

    def _reset(self):
        self.close()
        self.socket = zmq.Context.instance().socket(zmq.REQ)
        self.socket.linger = 0
        self.socket.sndtimeo = self.socket.rcvtimeo = self.timeout_ms
        self.socket.connect(self.endpoint)

    def request(self, cmd, **params):
        try:
            self.socket.send_json({"cmd": cmd, **params})
            reply = self.socket.recv_json()
        except Exception:
            self._reset()  # A timed-out REQ socket cannot send another request.
            raise
        if not isinstance(reply, dict) or not reply.get("ok"):
            raise RuntimeError(f"GX16 command failed: {reply}")
        return reply["result"]

    def get_qpos(self):
        return np.deg2rad(joint_values(self.request("getjs")["positions"]))

    def set_qpos(self, radians):
        return self.request(
            "setjs", positions=np.rad2deg(joint_values(radians)).tolist()
        )

    def close(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None
