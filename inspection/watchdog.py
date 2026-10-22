"""Independent command gate: a stopped/hung controller cannot sustain motion."""
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from inspection.core import watchdog_command


class Watchdog(Node):
    def __init__(self):
        super().__init__('inspection_watchdog')
        self.last = -float('inf')
        self.velocity = (0.0, 0.0)
        self.pub = self.create_publisher(Twist, '/turtle1/cmd_vel', 1)
        self.create_subscription(Twist, '/inspection/cmd_vel', self.receive, 1)
        self.create_timer(0.05, self.tick)

    def receive(self, msg):
        self.last = time.monotonic()
        self.velocity = (msg.linear.x, msg.angular.z)

    def tick(self):
        v, w = watchdog_command(*self.velocity, time.monotonic() - self.last)
        msg = Twist()
        msg.linear.x, msg.angular.z = v, w
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = Watchdog()
    try:
        rclpy.spin(node)
    finally:
        node.velocity = (0.0, 0.0)
        node.tick()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
