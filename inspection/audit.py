"""Verify recorded evidence, including real arrival tolerance and route consistency."""
import hashlib
import json
from pathlib import Path
import statistics
from inspection.render import load_evidence, read_jsonl
from inspection.core import decision, TARGETS


def main():
    run = Path(Path('output/latest_run.txt').read_text().strip())
    summary, events, poses = load_evidence(run)
    inferences = [e for e in events if e['event']=='inference']
    arrivals = [e for e in events if e['event']=='arrived']
    assert len(inferences)==len(arrivals)==len(summary['cases'])
    for infer, arrived, case in zip(inferences,arrivals,summary['cases']):
        assert infer['case_id']==arrived['case_id']==case['id']
        result=infer['response']
        assert hashlib.sha256((Path('data/images')/case['image']).read_bytes()).hexdigest()==result['sha256']
        label, reason = decision(result['result']['answers']['inspection'])
        assert infer['route']==label and infer['reason']==reason
        assert tuple(arrived['target'])==TARGETS[label]
        assert infer['time'] <= arrived['time']
        assert any(p['case_id']==case['id'] and p['phase']=='moving' for p in poses) or arrived['travel_seconds']<.1
        if reason=='low_confidence':
            stop=next(e for e in events if e['event']=='safety_stop' and e['case_id']==case['id'])
            stopped=[p for p in poses if stop['time']+.15<=p['time']<=stop['time']+.85]
            assert len(stopped)>=5 and all(abs(p['linear_velocity'])<.001 and abs(p['angular_velocity'])<.001 for p in stopped)
    safety=json.loads(Path('output/logs/safety_integration.json').read_text())
    assert len(safety)>=5 and all(x['passed'] for x in safety.values())
    frames=read_jsonl(run/'capture_timestamps.jsonl')
    assert len(frames)>100
    assert all(b['monotonic']>a['monotonic'] for a,b in zip(frames,frames[1:]))
    import cv2
    capture=cv2.VideoCapture(str(run/'turtlesim_raw.mp4'))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT))==len(frames)
    capture.release()
    results={'passed':True,'case_count':len(inferences),'distinct_routes':summary['distinct_routes'],
             'max_arrival_error':max(e['error'] for e in arrivals),
             'mean_inference_seconds':statistics.mean(c['inference_seconds'] for c in summary['cases']),
             'raw_accuracy':summary['raw_accuracy'],'pose_samples':len(poses),
             'route_accuracy':sum(c['route']==c['expected'] for c in summary['cases'])/len(summary['cases']),
             'low_confidence_stops':sum(c['reason']=='low_confidence' for c in summary['cases']),
             'max_peak_gpu_bytes':max(e['response']['peak_gpu_bytes'] for e in inferences),
             'actual_screen_frames':len(frames),
             'max_screen_frame_gap_seconds':max(b['monotonic']-a['monotonic'] for a,b in zip(frames,frames[1:])),
             'safety_checks':safety}
    (run/'audit.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
