"""Synthetic answers here are explicitly test fixtures, never inference evidence."""
import math
import pytest
from inspection.core import command, decision, watchdog_command, TARGETS, MonotonicEpoch


def answer(p):
    return {'choice': max(p, key=p.get), 'confidence': max(p.values()), 'probabilities': p}


def test_low_confidence_routes_recheck():
    assert decision(answer({'PASS': .4, 'REJECT': .35, 'RECHECK': .25})) == ('RECHECK', 'low_confidence')


def test_wall_clock_rollback_does_not_reorder_ros_log(monkeypatch):
    import inspection.core as core
    monkeypatch.setattr(core.time, 'time', lambda: 1000.)
    monkeypatch.setattr(core.time, 'monotonic', lambda: 10.)
    clock=MonotonicEpoch()
    monkeypatch.setattr(core.time, 'time', lambda: 900.)
    monkeypatch.setattr(core.time, 'monotonic', lambda: 11.5)
    assert clock.now()==1001.5


@pytest.mark.parametrize('p', [
    {'PASS': float('nan'), 'REJECT': .2, 'RECHECK': .2},
    {'PASS': .9, 'REJECT': .9, 'RECHECK': .2},
    {'PASS': 1., 'REJECT': 0.},
    {'PASS': 1.1, 'REJECT': -.1, 'RECHECK': 0.}])
def test_malformed_distributions_fail_closed(p):
    with pytest.raises(ValueError):
        decision({'choice': 'PASS', 'confidence': .9, 'probabilities': p})


def test_inconsistent_choice_rejected():
    a = answer({'PASS': .8, 'REJECT': .1, 'RECHECK': .1})
    a['choice'] = 'REJECT'
    with pytest.raises(ValueError):
        decision(a)


def test_watchdog_expiry_and_invalid_velocity():
    assert watchdog_command(1, 1, .51) == (0, 0)
    assert watchdog_command(float('nan'), 1, 0) == (0, 0)
    assert watchdog_command(1, 1, .1) == (1, 1)


def test_turn_before_forward():
    v, w, arrived = command((5, 5, math.pi), (8, 5))
    assert v == 0 and abs(w) > 0 and not arrived


def test_nonfinite_feedback_fails():
    with pytest.raises(ValueError):
        command((1, 1, float('nan')), (2, 2))


@pytest.mark.parametrize('initial', [(5.5, 5.5, 0), (1, 1, 3), (9, 9, -2)])
@pytest.mark.parametrize('target', list(TARGETS.values()))
def test_control_converges_under_kinematic_unit_model(initial, target):
    # Only a numerical unit test; actual ROS convergence is separately measured.
    x, y, th = initial
    for _ in range(3000):
        v, w, arrived = command((x, y, th), target)
        if arrived:
            break
        x += v*math.cos(th)*.02
        y += v*math.sin(th)*.02
        th += w*.02
    assert arrived
    assert math.hypot(x-target[0], y-target[1]) <= .12
