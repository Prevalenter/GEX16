"""GX16 hardware driver and command server."""

import argparse
import math
import signal
import threading
import time

import numpy as np
import zmq

import device
from transport import (
    COMMAND_ENDPOINT,
    GX16_ENDPOINT,
    GX16_TOPIC,
    joint_values,
    publish,
)

JOINT_DIRECTIONS = joint_values(device.config("gx16")["HAND"]["JOINT_MOTOR_DIRECTIONS"])
if not np.isin(JOINT_DIRECTIONS, [-1, 1]).all():
    raise ValueError("GX16 joint directions must be +1 or -1")


class Hand16:
    """Angles are motor-relative degrees, before the 90-degree hardware offset."""

    def __init__(self, port=None, serial_number=None):
        self.port = device.resolve_port(port, serial_number)
        self.handler, self.motors = None, []
        self.is_connected = False

    def connect(self, curr_limit=1000, goal_current=600, goal_pwm=200):
        if (
            not 0 < curr_limit <= 1750
            or not 0 <= goal_current <= curr_limit
            or not 0 <= goal_pwm <= 885
        ):
            raise ValueError("Invalid current/PWM limits")
        try:
            self.handler, self.motors = device.connect(
                self.port, device.config("gx16"), curr_limit
            )
            for motor in self.motors:
                motor.init_config(curr_limit, goal_current, goal_pwm)
            self.is_connected = True
        except BaseException:
            self.close()
            raise

    def off(self):
        for motor in self.motors:
            motor.torq_off()

    def on(self):
        for motor in self.motors:
            motor.torq_on()

    def getjs(self):
        if not self.is_connected:
            raise RuntimeError("GX16 is not connected")
        return joint_values([motor.get_pos() - 90 for motor in self.motors]).tolist()

    def setjs(self, angles):
        angles = joint_values(angles)
        if not self.is_connected:
            raise RuntimeError("GX16 is not connected")
        for motor, angle in zip(self.motors, angles):
            motor.set_pos(float(angle) + 90)

    def setj(self, joint, angle):
        if (
            isinstance(joint, bool)
            or not isinstance(joint, int)
            or not 1 <= joint <= 16
        ):
            raise ValueError("joint must be an integer in 1..16")
        if not math.isfinite(float(angle)):
            raise ValueError("position must be finite")
        if not self.is_connected:
            raise RuntimeError("GX16 is not connected")
        self.motors[joint - 1].set_pos(float(angle) + 90)

    def home(self):
        self.setjs([0] * 16)
        time.sleep(1)

    def close(self):
        try:
            device.close(self.handler, self.motors)
        finally:
            self.handler, self.motors = None, []
            self.is_connected = False


def handle_request(hand, request, stop):
    """Keep the original command/reply shape, without telemetry or simulation."""
    try:
        if not isinstance(request, dict):
            raise ValueError("Request must be a JSON object")
        cmd = request.get("cmd")
        units = request.get("units", "urdf_deg")
        if units not in ("urdf_deg", "motor_deg"):
            raise ValueError("units must be urdf_deg or motor_deg")
        direction = JOINT_DIRECTIONS if units == "urdf_deg" else np.ones(16)
        result = {}
        if cmd in ("ping", "status"):
            result = {"connected": hand.is_connected, "port": hand.port}
        elif cmd == "getjs":
            result = {
                "positions": (joint_values(hand.getjs()) * direction).tolist(),
                "units": units,
            }
        elif cmd == "setjs":
            hand.setjs(joint_values(request.get("positions")) * direction)
        elif cmd == "setj":
            joint = request.get("joint")
            if (
                isinstance(joint, bool)
                or not isinstance(joint, int)
                or not 1 <= joint <= 16
            ):
                raise ValueError("joint must be an integer in 1..16")
            angle = float(request["position"])
            if not math.isfinite(angle):
                raise ValueError("position must be finite")
            hand.setj(joint, angle * direction[joint - 1])
        elif cmd in ("torque_on", "torque_off", "home"):
            {"torque_on": hand.on, "torque_off": hand.off, "home": hand.home}[cmd]()
        elif cmd == "shutdown":
            stop.set()
        else:
            raise ValueError(f"Unknown command: {cmd!r}")
        return {"ok": True, "result": result, "error": None}
    except Exception as exc:
        return {"ok": False, "result": None, "error": str(exc)}


def run(args, hand=None, stop=None):
    hand = hand or Hand16(args.port, args.serial_number)
    stop = stop or threading.Event()
    context = zmq.Context.instance()
    commands, states = context.socket(zmq.REP), context.socket(zmq.PUB)
    commands.linger = states.linger = 0
    try:
        commands.bind(args.cmd_endpoint)
        states.bind(args.state_endpoint)
        hand.connect(args.curr_limit, args.goal_current, args.goal_pwm)
        print(
            f"GX16 {hand.port}: commands={args.cmd_endpoint}, state={args.state_endpoint}",
            flush=True,
        )
        next_state = 0
        # A single loop owns the serial port and both ZMQ sockets.
        while not stop.is_set():
            if commands.poll(5):
                try:
                    reply = handle_request(hand, commands.recv_json(), stop)
                except ValueError as exc:
                    reply = {"ok": False, "result": None, "error": str(exc)}
                commands.send_json(reply)
            if not stop.is_set() and time.monotonic() >= next_state:
                angles = joint_values(hand.getjs())
                publish(
                    states,
                    GX16_TOPIC,
                    {
                        "timestamp": time.time(),
                        "motor_deg": angles.tolist(),
                        "urdf_deg": (angles * JOINT_DIRECTIONS).tolist(),
                    },
                )
                next_state = time.monotonic() + 1 / args.state_hz
    finally:
        commands.close()
        states.close()
        hand.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    connection = parser.add_mutually_exclusive_group(required=True)
    connection.add_argument("--port")
    connection.add_argument("--serial-number")
    parser.add_argument("--cmd-endpoint", default=COMMAND_ENDPOINT)
    parser.add_argument("--state-endpoint", default=GX16_ENDPOINT)
    parser.add_argument("--state-hz", type=float, default=10)
    parser.add_argument("--curr-limit", type=int, default=1000)
    parser.add_argument("--goal-current", type=int, default=600)
    parser.add_argument("--goal-pwm", type=int, default=200)
    args = parser.parse_args()
    if not math.isfinite(args.state_hz) or args.state_hz <= 0:
        parser.error("--state-hz must be finite and positive")
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    run(args, stop=stop)


if __name__ == "__main__":
    main()
