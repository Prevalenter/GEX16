"""Serial setup shared by the two hardware drivers."""

import yaml
from serial.tools.list_ports import comports

from _vendor.dynamixel_sdk import PacketHandler, PortHandler
from motor import Motor
from paths import RESOURCE_ROOT


def config(name):
    return yaml.safe_load((RESOURCE_ROOT / name / "config.yaml").read_text())


def resolve_port(port, serial_number):
    if port:
        return port
    if not serial_number:
        raise ValueError("port or serial_number is required")
    matches = [p.device for p in comports() if p.serial_number == serial_number]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one device for serial {serial_number!r}, found {len(matches)}"
        )
    return matches[0]


def connect(port, settings, curr_max=1750):
    handler = PortHandler(port)
    try:
        if not handler.openPort() or not handler.setBaudRate(
            settings["BASIC"]["BAUDRATE"]
        ):
            raise RuntimeError(f"Cannot open serial port {port}")
        packet = PacketHandler(settings["BASIC"]["PROTOCOL_VERSION"])
        motors = [Motor(i, handler, packet, curr_max) for i in range(1, 17)]
        return handler, motors
    except BaseException:
        handler.closePort()
        raise


def close(handler, motors):
    """Attempt torque-off on every motor even if one operation fails."""
    error = None
    try:
        for motor in motors:
            try:
                motor.torq_off()
            except Exception as exc:
                error = error or exc
    finally:
        if handler is not None:
            handler.closePort()
    if error is not None:
        raise error
