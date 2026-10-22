"""Create a local, privacy-conscious submission ZIP. Never uploads or publishes."""
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    run=Path(Path('output/latest_run.txt').read_text().strip())
    audit=json.loads((run/'audit.json').read_text())
    assert audit['passed']
    dest=Path('output/submission/physicalai_source_and_evidence.zip')
    paths=[]
    for directory in ['inspection','scripts','tests','data','docs']:
        paths.extend(p for p in Path(directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths += [Path(n) for n in ['README.md','Dockerfile','compose.yaml','requirements.txt',
                              'requirements.lock.txt','.env','.gitignore','.gitattributes','.dockerignore']]
    for name in ['summary.json','events.jsonl','poses.jsonl','ros_graph.json','audit.json','turtlesim_raw.mp4','capture_timestamps.jsonl']:
        paths.append(run/name)
    for name in ['safety_integration.json','unit_tests.xml','unit_tests.txt','model_metadata.json','image_provenance.json']:
        paths.append(Path('output/logs')/name)
    if Path('output/logs/prompt_experiments.json').exists():
        paths.append(Path('output/logs/prompt_experiments.json'))
    paths += [Path('output/latest_run.txt'),Path('output/submission/video_metadata.json'),Path('output/submission/report.md')]
    manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p,p.as_posix())
        z.writestr('MANIFEST.sha256.json',json.dumps(manifest,indent=2))
    print(dest, dest.stat().st_size)


if __name__=='__main__':
    main()
