"""Extract Japanese media QA frames and verify measured evidence is unchanged."""
import hashlib
import json
from pathlib import Path
import subprocess

qa=Path('output/qa')
before=json.loads((qa/'japanese_source_hashes.json').read_text())
assert all(hashlib.sha256(Path(name.replace('\\','/')).read_bytes()).hexdigest()==digest for name,digest in before.items())
run=Path(Path('output/latest_run.txt').read_text().strip())
events=[json.loads(s) for s in (run/'events.jsonl').read_text().splitlines()]
start=next(e['time'] for e in events if e['event']=='run_start')
stops=[e['time']-start+6.5 for e in events if e['event']=='safety_stop']
for tag,seconds in [(str(x),x) for x in [2,12,28,45,66,82,87]]+[(f'stop_{i}',t) for i,t in enumerate(stops,1)]:
    subprocess.run(['ffmpeg','-y','-loglevel','error','-ss',str(seconds),'-i',
                    'output/submission/inspection_demo.mp4','-frames:v','1',str(qa/f'japanese-video_{tag}.png')],check=True)
print('Japanese video QA frames extracted; all original measured evidence hashes unchanged.')
