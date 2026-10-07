"""EX16 hardware reader and joint-state publisher."""

import argparse
import math
import signal
import threading
import time

import numpy as np
import zmq

import device
from transport import EX16_ENDPOINT, EX16_TOPIC, joint_values, publish


class Glove16:
    """Read URDF joint angles in degrees; the glove remains torque-disabled."""

    def __init__(self, port=None, serial_number=None, left=False):
        self.port = device.resolve_port(port, serial_number)
        self.settings = device.config("ex16")
        self.directions = joint_values(self.settings["HAND"]["JOINT_MOTOR_DIRECTIONS"])
        if not np.isin(self.directions, [-1, 1]).all():
            raise ValueError("EX16 joint directions must be +1 or -1")
        if left:
            self.directions *= [
                -1,
                -1,
                -1,
                -1,
                1,
                -1,
                -1,
                -1,
                1,
                -1,
                -1,
                -1,
                1,
                -1,
                -1,
                -1,
            ]
        self.handler, self.motors = None, []
        self.is_connected = False

    def connect(self):
        try:
            self.handler, self.motors = device.connect(self.port, self.settings)
            angles = np.array([motor.get_pos() for motor in self.motors])
            self.offsets = np.where(angles < 270, 0, 360)
            self.off()
            self.is_connected = True
        except BaseException:
            self.close()
            raise

    def getjs(self):
        if not self.is_connected:
            raise RuntimeError("EX16 is not connected")
        angles = joint_values([motor.get_pos() for motor in self.motors])
        return (angles - 90 - self.offsets) * self.directions

    def off(self):
        for motor in self.motors:
            motor.torq_off()

    def close(self):
        try:
            device.close(self.handler, self.motors)
        finally:
            self.handler, self.motors = None, []
            self.is_connected = False


def run(args, glove=None, stop=None):
    glove = glove or Glove16(args.port, args.serial_number, args.left)
    stop = stop or threading.Event()
    publisher = zmq.Context.instance().socket(zmq.PUB)
    publisher.linger = 0
    try:
        # Fail on an occupied endpoint before opening hardware.
        publisher.bind(args.state_endpoint)
        glove.connect()
        print(f"EX16 {glove.port} -> {args.state_endpoint}", flush=True)
        sequence = 0
        while not stop.is_set():
            started = time.monotonic()
            publish(
                publisher,
                EX16_TOPIC,
                {
                    "name": "ex16",
                    "sequence": sequence,
                    "timestamp": time.time(),
                    "urdf_deg": joint_values(glove.getjs()).tolist(),
                },
            )
            sequence += 1
            stop.wait(max(0, 1 / args.state_hz - (time.monotonic() - started)))
    finally:
        publisher.close()
        glove.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    connection = parser.add_mutually_exclusive_group(required=True)
    connection.add_argument("--port")
    connection.add_argument("--serial-number")
    parser.add_argument("--left", action="store_true")
    parser.add_argument("--state-endpoint", default=EX16_ENDPOINT)
    parser.add_argument("--state-hz", type=float, default=100)
    args = parser.parse_args()
    if not math.isfinite(args.state_hz) or args.state_hz <= 0:
        parser.error("--state-hz must be finite and positive")
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    run(args, stop=stop)


if __name__ == "__main__":
    main()
