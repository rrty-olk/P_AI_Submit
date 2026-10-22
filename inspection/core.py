"""Pure control policy; no ROS/model dependencies, no simulated model answers."""
import math
import time

LABELS = ('PASS', 'REJECT', 'RECHECK')
TARGETS = {'PASS': (8.5, 8.5), 'REJECT': (2.0, 8.5), 'RECHECK': (5.5, 2.0)}
MIN_CONFIDENCE = 0.60


class MonotonicEpoch:
    """Epoch-labelled run time immune to WSL/NTP wall-clock corrections."""
    def __init__(self):
        self.wall_origin = time.time()
        self.mono_origin = time.monotonic()

    def now(self):
        return self.wall_origin + time.monotonic() - self.mono_origin


def decision(answer, threshold=MIN_CONFIDENCE):
    p = answer['probabilities']
    if set(p) != set(LABELS):
        raise ValueError('Missing or unknown probability labels')
    if not all(isinstance(v, (float, int)) and math.isfinite(v) and 0 <= v <= 1 for v in p.values()):
        raise ValueError('Invalid probability')
    if abs(sum(p.values()) - 1) > 1e-4:
        raise ValueError('Probabilities do not sum to one')
    top = max(p, key=p.get)
    confidence = answer['confidence']
    if answer['choice'] != top or not math.isfinite(confidence) or abs(confidence - p[top]) > 1e-4:
        raise ValueError('Inconsistent model answer')
    # Low confidence always stops first, then routes to human reinspection.
    return ('RECHECK', 'low_confidence') if confidence < threshold else (top, 'model')


def command(pose, target, tolerance=0.12):
    x, y, theta = pose
    if not all(math.isfinite(v) for v in (*pose, *target)):
        raise ValueError('Non-finite pose/target')
    distance = math.hypot(target[0] - x, target[1] - y)
    if distance <= tolerance:
        return 0.0, 0.0, True
    error = math.atan2(math.sin(math.atan2(target[1] - y, target[0] - x) - theta),
                       math.cos(math.atan2(target[1] - y, target[0] - x) - theta))
    angular = max(-2.5, min(2.5, 4.0 * error))
    linear = min(1.7, 1.2 * distance) * max(0.0, math.cos(error)) if abs(error) < 0.65 else 0.0
    return linear, angular, False


def watchdog_command(linear, angular, age, timeout=0.5):
    if age < 0 or age > timeout or not all(math.isfinite(v) for v in (linear, angular)):
        return 0.0, 0.0
    return max(0.0, min(1.7, linear)), max(-2.5, min(2.5, angular))
