"""Development experiment with the real fixed model, preserving all outcomes.

Used to investigate initial all-PASS failure. Never supplies expected labels.
"""
import json
import os
from pathlib import Path
import time
import torch
from PIL import Image
from transformers import AutoModel
from huggingface_hub import snapshot_download

prompts = [
    {'type':'choice','instructions':'Describe the visible shape in this image.',
     'criteria':{'PASS':'A complete circular gray ring with a hole.',
                 'REJECT':'A broken C-shaped gray ring with a missing piece.',
                 'RECHECK':'A dark rectangle obscures the gray ring.'}},
    {'type':'choice','instructions':'Is the metal ring complete, broken, or hidden?',
     'criteria':{'PASS':'Complete metal ring.', 'REJECT':'Broken metal ring.', 'RECHECK':'Hidden metal ring.'}},
    {'type':'choice','instructions':'Choose the most accurate description of the image.',
     'criteria':{'PASS':'An intact washer.', 'REJECT':'A damaged washer.', 'RECHECK':'An obscured washer.'}},
    {'type':'choice','instructions':'What shape is visible?',
     'criteria':{'PASS':'A full circular ring.', 'REJECT':'An open ring with a gap.', 'RECHECK':'A black rectangle.'}},
]
path=snapshot_download('LiquidAI/d1-omni-600M',revision=os.environ['MODEL_REVISION'])
model=AutoModel.from_pretrained(path,trust_remote_code=True,dtype=torch.float16).to('cuda').eval()
_=model.tokenizer
results=[]
for pi,q in enumerate(prompts):
    for name in ['01_intact.png','02_broken.png','03_covered.png']:
        image=Image.open(Path('data/images')/name).convert('RGB')
        torch.cuda.synchronize()
        start=time.perf_counter()
        with torch.inference_mode():
            result=model.system_one(None,{'inspection':q},images=[image])
        torch.cuda.synchronize()
        item={'prompt_index':pi,'question':q,'image':name,'result':result,
              'inference_seconds':time.perf_counter()-start,'revision':os.environ['MODEL_REVISION']}
        results.append(item)
        print(json.dumps(item),flush=True)
Path('output/logs/prompt_experiments.json').write_text(json.dumps(results,indent=2))
