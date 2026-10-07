"""Dynamixel operations used by EX16/GX16. Register values match the hardware config."""

import time


class Motor:
    def __init__(self, motor_id, port, packet, curr_max=1750):
        self.id, self.port, self.packet = motor_id, port, packet
        self.curr_max = curr_max
        self.unit_scale = 0.087891

    def torq_on(self):
        self.packet.write1ByteTxRx(self.port, self.id, 64, 1)

    def torq_off(self):
        self.packet.write1ByteTxRx(self.port, self.id, 64, 0)

    def get_pos(self):
        position, _, _ = self.packet.read4ByteTxRx(self.port, self.id, 132)
        if position >= 2**31:
            position -= 2**32
        return position * self.unit_scale

    def set_pos(self, degrees):
        self.packet.write4ByteTxRx(
            self.port, self.id, 116, int(degrees / self.unit_scale)
        )

    def init_config(self, curr_limit=1000, goal_current=600, goal_pwm=200):
        for _ in range(2):
            for led in (1, 0):
                self.packet.write1ByteTxRx(self.port, self.id, 65, led)
                time.sleep(0.02)
        self.torq_off()
        limit = min(curr_limit, self.curr_max)
        self.packet.write2ByteTxRx(self.port, self.id, 38, limit)
        self.packet.write1ByteTxRx(self.port, self.id, 11, 5)
        self.packet.write2ByteTxRx(self.port, self.id, 102, min(goal_current, limit))
        self.packet.write2ByteTxRx(self.port, self.id, 100, min(goal_pwm, 885))
        self.torq_on()
