"""GPU-only official d1 system_one inference. No fallback model or dummy responses."""
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import transformers
from transformers import AutoModel
from huggingface_hub import snapshot_download
from PIL import Image

from inspection.core import decision

MODEL_ID = 'LiquidAI/d1-omni-600M'
QUESTION = {'inspection': {
    'type': 'choice',
    'instructions': 'Choose the most accurate description of the image.',
    'criteria': {
        'PASS': 'An intact washer.',
        'REJECT': 'A damaged washer.',
        'RECHECK': 'An obscured washer.'}}}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is mandatory; refusing CPU or fabricated inference')
    revision = os.environ['MODEL_REVISION']
    if len(revision) != 40:
        raise ValueError('Require immutable 40-character model commit')
    path = snapshot_download(MODEL_ID, revision=revision)
    model = AutoModel.from_pretrained(path, trust_remote_code=True, dtype=torch.float16).to('cuda').eval()
    # Resolve tokenizer from the same downloaded snapshot, not mutable main.
    _ = model.tokenizer
    meta = {'model_id': MODEL_ID, 'revision': revision, 'device': str(model.device),
            'dtype': str(next(model.parameters()).dtype), 'gpu': torch.cuda.get_device_name(0),
            'torch': torch.__version__, 'transformers': transformers.__version__,
            'cuda_runtime': torch.version.cuda, 'api': 'system_one', 'question': QUESTION,
            'parameter_count': sum(p.numel() for p in model.parameters())}
    Path('output/logs').mkdir(parents=True, exist_ok=True)
    Path('output/logs/model_metadata.json').write_text(json.dumps(meta, indent=2))
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, obj):
            data = json.dumps(obj, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.send(200 if self.path == '/health' else 404, meta)

        def do_POST(self):
            try:
                if self.path != '/infer':
                    return self.send(404, {'error': 'Unknown endpoint'})
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096:
                    raise ValueError('Invalid request size')
                request = json.loads(self.rfile.read(length))
                root = Path('data/images').resolve()
                image_path = (root / request['image']).resolve()
                if image_path.parent != root or image_path.suffix.lower() != '.png':
                    raise ValueError('Image must be a PNG directly under data/images')
                digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
                with Image.open(image_path) as img:
                    image = img.convert('RGB')
                if not lock.acquire(blocking=False):
                    return self.send(503, {'error': 'Inference busy'})
                try:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    with torch.inference_mode():
                        result = model.system_one(None, QUESTION, images=[image])
                    torch.cuda.synchronize()
                    duration = time.perf_counter() - start
                    decision(result['answers']['inspection'])
                    response = {'result': result, 'inference_seconds': duration,
                                'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
                                'image': image_path.name, 'sha256': digest, 'metadata': meta,
                                'request_id': request['request_id'], 'time': time.time()}
                    with open('output/logs/inference.jsonl', 'a') as f:
                        f.write(json.dumps(response) + '\n')
                    self.send(200, response)
                finally:
                    lock.release()
            except Exception as exc:
                self.send(500, {'error': f'{type(exc).__name__}: {exc}'})

    print(json.dumps(meta), flush=True)
    ThreadingHTTPServer(('0.0.0.0', 8000), Handler).serve_forever()


if __name__ == '__main__':
    main()
