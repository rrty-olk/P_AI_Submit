"""Deterministic synthetic input photographs, NOT model predictions.

Ground truth is for evaluation only and never sent to the model.
"""
import json
import math
from pathlib import Path
import random
from PIL import Image, ImageDraw, ImageFilter


def main():
    root = Path('data/images')
    root.mkdir(parents=True, exist_ok=True)
    cases = []
    for i, kind in enumerate(['intact', 'broken', 'covered', 'broken', 'intact', 'covered',
                               'intact', 'broken', 'covered']):
        rng = random.Random(7300 + i)
        im = Image.new('RGB', (512, 512), (225, 224, 217))
        d = ImageDraw.Draw(im)
        cx, cy, r = 256 + rng.randint(-15, 15), 256 + rng.randint(-15, 15), 155
        d.ellipse((cx-r+8, cy-r+12, cx+r+8, cy+r+12), fill=(164, 164, 158))
        d.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(113, 125, 137), outline=(63, 72, 80), width=4)
        for j in range(10):
            rr = r - 6 - j*2
            d.arc((cx-rr, cy-rr, cx+rr, cy+rr), 195, 315, fill=(185-j*4, 193-j*4, 201-j*4), width=2)
        inner = 72
        d.ellipse((cx-inner, cy-inner, cx+inner, cy+inner), fill=(225, 224, 217), outline=(65, 70, 73), width=5)
        if kind == 'broken':
            # Large missing wedge makes physical damage unambiguous to a human.
            angle = [-0.2, 0.7, 2.5][i % 3]
            points = [(cx, cy)] + [(cx + 210*math.cos(a), cy + 210*math.sin(a))
                                          for a in (angle-0.23, angle+0.23)]
            d.polygon(points, fill=(225, 224, 217))
        if kind == 'covered':
            d.rectangle((70, 125, 450, 365), fill=(37, 41, 47))
            im = im.filter(ImageFilter.GaussianBlur(6))
        name = f'{i+1:02d}_{kind}.png'
        im.save(root / name)
        cases.append({'id': f'case_{i+1:02d}', 'image': name,
                      'expected': {'intact': 'PASS', 'broken': 'REJECT', 'covered': 'RECHECK'}[kind],
                      'source': 'deterministically generated synthetic washer inspection image'})
    Path('data/cases.json').write_text(json.dumps(cases, indent=2))
    print(f'Generated {len(cases)} synthetic image inputs; no inference performed.')


if __name__ == '__main__':
    main()
