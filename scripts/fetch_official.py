"""Archive exact official model source and revision before enabling remote code."""
import json
import pathlib
import urllib.request

out = pathlib.Path('docs/official_model')
out.mkdir(parents=True, exist_ok=True)
base = 'https://huggingface.co/'
with urllib.request.urlopen(base + 'api/models/LiquidAI/d1-omni-600M', timeout=60) as r:
    meta = json.load(r)
(out / 'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
revision = meta['sha']
for entry in meta['siblings']:
    name = entry['rfilename']
    if name.endswith('.py') or name in ('config.json', 'README.md', 'LICENSE'):
        with urllib.request.urlopen(base + f'LiquidAI/d1-omni-600M/resolve/{revision}/{name}', timeout=60) as r:
            target = out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(r.read())
pathlib.Path('.env').write_text(f'MODEL_REVISION={revision}\n', encoding='utf-8')
print('Pinned official model revision:', revision)
