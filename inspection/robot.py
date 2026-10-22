"""Real turtlesim feedback loop; aborts the run on any transport/control fault."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

import requests
import rclpy
from rclpy.node import Node
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist

from inspection.core import command, decision, TARGETS, MonotonicEpoch


class Robot(Node):
    def __init__(self, run):
        super().__init__('inspection_controller')
        self.run = run
        self.event_clock = MonotonicEpoch()
        self.events = open(run / 'events.jsonl', 'w', buffering=1)
        self.poses = open(run / 'poses.jsonl', 'w', buffering=1)
        self.pose = None
        self.last_pose = 0
        self.case_id = None
        self.phase = 'startup'
        self.pub = self.create_publisher(Twist, '/inspection/cmd_vel', 1)
        self.create_subscription(Pose, '/turtle1/pose', self.receive, 10)
        self.last_record = 0

    def receive(self, msg):
        self.pose = (float(msg.x), float(msg.y), float(msg.theta))
        self.last_pose = time.monotonic()
        if self.last_pose - self.last_record >= 0.049:
            self.poses.write(json.dumps({'time': self.event_clock.now(), 'x': msg.x, 'y': msg.y,
                                        'theta': msg.theta, 'linear_velocity': msg.linear_velocity,
                                        'angular_velocity': msg.angular_velocity,
                                        'case_id': self.case_id, 'phase': self.phase}) + '\n')
            self.last_record = self.last_pose

    def event(self, name, **kwargs):
        item = {'time': self.event_clock.now(), 'event': name, 'case_id': self.case_id, **kwargs}
        self.events.write(json.dumps(item, allow_nan=False) + '\n')
        print(json.dumps(item), flush=True)

    def velocity(self, v=0.0, w=0.0):
        msg = Twist()
        msg.linear.x, msg.angular.z = float(v), float(w)
        self.pub.publish(msg)

    def tick(self):
        rclpy.spin_once(self, timeout_sec=0.02)
        if not rclpy.ok():
            raise RuntimeError('ROS shutdown')

    def fresh(self):
        if self.pose is None or time.monotonic() - self.last_pose > 0.5:
            self.velocity()
            raise TimeoutError('Pose feedback missing/stale > 0.5s')
        if not all(math.isfinite(p) for p in self.pose):
            self.velocity()
            raise ValueError('Invalid pose feedback')

    def hold(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.tick()
            self.fresh()
            self.velocity()
            time.sleep(0.02)

    def await_inference(self, future, deadline=30):
        start = time.monotonic()
        while not future.done():
            self.tick()
            self.fresh()
            self.velocity()
            if time.monotonic()-start > deadline:
                raise TimeoutError('Inference deadline exceeded')
            time.sleep(0.02)
        return future.result()

    def move(self, target, timeout=30):
        self.phase = 'moving'
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            self.tick()
            self.fresh()
            v, w, reached = command(self.pose, target)
            self.velocity(v, w)
            if reached:
                self.event('arrived', target=target, pose=self.pose,
                           error=math.hypot(target[0]-self.pose[0], target[1]-self.pose[1]),
                           travel_seconds=time.monotonic()-start)
                self.phase = 'arrived'
                self.hold(1.5)
                return
            time.sleep(0.02)
        self.velocity()
        raise TimeoutError('Target not reached within 30s')


def main():
    run = Path('output/runs') / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:6])
    run.mkdir(parents=True)
    Path('output/latest_run.txt').write_text(str(run))
    rclpy.init()
    node = Robot(run)
    pool = ThreadPoolExecutor(max_workers=1)
    recorder = None
    summary = {'status': 'failed', 'run': run.name, 'cases': [], 'start_time': node.event_clock.now(),
               'time_base': 'epoch origin + monotonic elapsed (immune to wall-clock steps)',
               'clock_wall_origin': node.event_clock.wall_origin,
               'clock_monotonic_origin': node.event_clock.mono_origin}
    try:
        deadline = time.monotonic() + 20
        while node.pose is None and time.monotonic() < deadline:
            node.tick()
            node.velocity()
        node.fresh()
        node.hold(1)
        if node.pub.get_subscription_count() < 1:
            raise RuntimeError('Watchdog command subscriber missing')
        graph = dict(node.get_topic_names_and_types())
        if graph.get('/turtle1/pose') != ['turtlesim/msg/Pose']:
            raise RuntimeError('Unexpected pose interface')
        (run / 'ros_graph.json').write_text(json.dumps(graph, indent=2))
        recorder = subprocess.Popen(['python','-m','inspection.capture',str(run)],
                     stderr=open(run/'recorder.log', 'w'))
        meta = requests.get(os.environ.get('AI_URL', 'http://ai:8000') + '/health', timeout=5)
        meta.raise_for_status()
        metadata = meta.json()
        if metadata['model_id'] != 'LiquidAI/d1-omni-600M' or metadata['device'] != 'cuda:0':
            raise RuntimeError('GPU 600M model provenance mismatch')
        summary['model_metadata'] = metadata
        node.event('run_start', model_metadata=metadata, capture_time=node.event_clock.now())
        cases = json.loads(Path('data/cases.json').read_text())
        for case in cases:
            node.case_id = case['id']
            node.phase = 'inspecting'
            node.event('case_start', image=case['image'], expected=case['expected'])
            node.hold(2)
            request_id = uuid.uuid4().hex
            future = pool.submit(requests.post, os.environ.get('AI_URL', 'http://ai:8000') + '/infer',
                                 json={'image': case['image'], 'request_id': request_id}, timeout=(5, 25))
            response = node.await_inference(future)
            response.raise_for_status()
            result = response.json()
            digest = hashlib.sha256((Path('data/images')/case['image']).read_bytes()).hexdigest()
            if result['request_id'] != request_id or result['sha256'] != digest:
                raise ValueError('Inference/image identity mismatch')
            answer = result['result']['answers']['inspection']
            route, reason = decision(answer)
            node.event('inference', response=result, route=route, reason=reason)
            if reason == 'low_confidence':
                node.phase = 'low_confidence_stop'
                node.event('safety_stop', reason=reason)
                node.hold(1)
            node.move(TARGETS[route])
            summary['cases'].append({**case, 'model_choice': answer['choice'], 'route': route,
                                     'reason': reason, 'confidence': answer['confidence'],
                                     'probabilities': answer['probabilities'],
                                     'inference_seconds': result['inference_seconds']})
        summary['distinct_routes'] = len({c['route'] for c in summary['cases']})
        summary['raw_accuracy'] = sum(c['model_choice']==c['expected'] for c in summary['cases']) / len(cases)
        if summary['distinct_routes'] < 2:
            raise RuntimeError('No destination changes; inspection demonstration is incomplete')
        summary['status'] = 'complete'
        node.event('run_complete', cases=len(cases), distinct_routes=summary['distinct_routes'])
    except BaseException as exc:
        summary['status'] = 'failed'
        summary['error'] = f'{type(exc).__name__}: {exc}'
        node.event('fault', error=summary['error'])
        raise
    finally:
        node.velocity()
        for _ in range(10):
            if rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.02)
                node.velocity()
        summary['end_time'] = node.event_clock.now()
        (run/'summary.json').write_text(json.dumps(summary, indent=2))
        if recorder:
            recorder.send_signal(signal.SIGINT)
            recorder.wait(timeout=15)
        pool.shutdown(wait=False, cancel_futures=True)
        node.events.close()
        node.poses.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
