"""Fault injection against real turtlesim and the separate watchdog."""
import json
from pathlib import Path
import time
from concurrent.futures import Future
import requests
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose


def main():
    rclpy.init()
    node = Node('safety_integration_test')
    pub = node.create_publisher(Twist, '/inspection/cmd_vel', 1)
    poses = []
    node.create_subscription(Pose, '/turtle1/pose', lambda m: poses.append((time.monotonic(), m)), 10)

    def spin(seconds, velocity=None):
        end = time.monotonic()+seconds
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=.02)
            if velocity is not None:
                msg = Twist()
                msg.linear.x = velocity
                pub.publish(msg)
            time.sleep(.02)

    results = {}
    try:
        spin(2, 0.)
        assert poses and pub.get_subscription_count() >= 1
        spin(.8, 1.)
        assert any(p.linear_velocity > .5 for _, p in poses[-8:]), 'No initial real motion'
        spin(1.)  # cease upstream commands, simulating hung/dead controller
        last = poses[-1][1]
        results['watchdog_expiry'] = {'passed': abs(last.linear_velocity) < .001,
                                       'velocity_after_1s': last.linear_velocity}
        assert results['watchdog_expiry']['passed']
        spin(.8, float('nan'))
        last = poses[-1][1]
        results['invalid_command'] = {'passed': abs(last.linear_velocity)<.001,
                                     'velocity': last.linear_velocity}
        assert results['invalid_command']['passed']
        response = requests.post('http://ai:8000/infer',
                                 json={'image':'missing.png', 'request_id':'fault-test'}, timeout=10)
        results['missing_image'] = {'passed': response.status_code==500, 'status':response.status_code}
        assert results['missing_image']['passed']
        # Exercise real controller fault handlers with genuine ROS feedback.
        from inspection.robot import Robot
        fault_dir = Path('output/safety_controller')
        fault_dir.mkdir(exist_ok=True)
        controller = Robot(fault_dir)
        controller.pose = (5., 5., 0.)
        controller.last_pose = time.monotonic()-1
        try:
            controller.fresh()
            raise AssertionError('Stale feedback accepted')
        except TimeoutError:
            results['stale_feedback'] = {'passed': True}
        # Actual /pose feedback remains alive; unreachable target triggers deadline.
        controller.tick()
        try:
            controller.move((100., 100.), timeout=.3)
            raise AssertionError('Unreachable goal accepted')
        except TimeoutError:
            results['movement_timeout'] = {'passed': True}
        # A never-completing future injects a transport hang, NOT an AI answer.
        try:
            controller.await_inference(Future(), deadline=.3)
            raise AssertionError('Hung inference accepted')
        except TimeoutError:
            results['inference_timeout'] = {'passed': True}
        controller.velocity()
        controller.events.close()
        controller.poses.close()
        controller.destroy_node()
        spin(.8, 0.)
    finally:
        spin(.5, 0.)
        Path('output/logs/safety_integration.json').write_text(json.dumps(results, indent=2))
        node.destroy_node()
        rclpy.shutdown()
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
