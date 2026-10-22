"""Render a dashboard from actual ROS telemetry and official GPU inference logs.

No kinematic integration or expected-label substitution is permitted here.
"""
import argparse
import bisect
import json
import math
from pathlib import Path
import statistics
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from inspection.core import TARGETS
from inspection.localization import LABEL_JA, FONT_REGULAR

COLORS = {'PASS': '#49d6a2', 'REJECT': '#ff6f7e', 'RECHECK': '#ffc767'}
FONT = str(FONT_REGULAR)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_evidence(run):
    summary = json.loads((run/'summary.json').read_text())
    events = read_jsonl(run/'events.jsonl')
    poses = read_jsonl(run/'poses.jsonl')
    if any(b['time'] < a['time'] for rows in (events,poses) for a,b in zip(rows,rows[1:])):
        raise ValueError('Non-monotonic telemetry timestamps; repeat run using monotonic clock')
    if summary['status'] != 'complete' or len(summary['cases']) < 3:
        raise ValueError('Refusing a completion video from an incomplete run')
    if summary.get('distinct_routes', 0) < 2:
        raise ValueError('No demonstrated destination change; refusing completion video')
    meta = summary['model_metadata']
    if meta['model_id'] != 'LiquidAI/d1-omni-600M' or meta['device'] != 'cuda:0' or meta['api'] != 'system_one':
        raise ValueError('No real official GPU inference evidence')
    if not poses or not (run/'ros_graph.json').exists():
        raise ValueError('No ROS pose/graph evidence')
    arrivals = [e for e in events if e['event']=='arrived']
    if len(arrivals)!=len(summary['cases']) or any(e['error'] > .12 for e in arrivals):
        raise ValueError('Goal feedback validation failed')
    return summary, events, poses


def video(run, dest):
    summary, events, poses = load_evidence(run)
    start = next(e['time'] for e in events if e['event']=='run_start')
    finish = next(e['time'] for e in events if e['event']=='run_complete')
    elapsed = finish-start
    speed = 1 if elapsed <= 160 else 2
    if elapsed/speed > 164:
        raise ValueError('Run too long for 3 minute video even at permitted 2x speed')
    intro, outro = 6., max(6., 61.-6.-elapsed/speed)
    duration = intro + elapsed/speed + outro
    fps, width, height = 15, 1280, 720
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(fps), '-i', '-', '-an',
               '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(dest)], stdin=subprocess.PIPE)
    fonts = {s: ImageFont.truetype(FONT, s) for s in [14, 16, 18, 21, 25, 32, 40]}
    pose_times = [p['time'] for p in poses]
    case_events = [e for e in events if e['event']=='case_start']
    inference_events = [e for e in events if e['event']=='inference']
    arrivals = [e for e in events if e['event']=='arrived']
    images = {e['image']: Image.open(Path('data/images')/e['image']).convert('RGB').resize((310,310))
              for e in case_events}
    capture = cv2.VideoCapture(str(run/'turtlesim_raw.mp4'))
    capture_times = [f['monotonic'] for f in read_jsonl(run/'capture_timestamps.jsonl')]
    if not capture_times or any(b<a for a,b in zip(capture_times,capture_times[1:])):
        raise ValueError('Invalid actual-frame timestamp evidence')
    raw_frame = None
    raw_index = -1

    def text(draw, xy, value, size=18, fill='#e8eef7', max_width=None):
        if max_width is None:
            x=xy[0]
            max_width=(347-x if 37<=x<365 else 887-x if 399<=x<905 else 1242-x if x>=938 else width-x-20)
        while True:
            if size not in fonts:
                fonts[size]=ImageFont.truetype(FONT,size)
            if draw.textlength(str(value),font=fonts[size])<=max_width or size<=12:
                break
            size-=1
        draw.text(xy, str(value), font=fonts[size], fill=fill)

    for frame_idx in range(math.ceil(duration*fps)):
        vt = frame_idx/fps
        t = start + max(0, min(elapsed, (vt-intro)*speed))
        canvas = Image.new('RGB', (width,height), '#0b1220')
        d = ImageDraw.Draw(canvas)
        text(d, (28,18), '画像検査AIとROS 2による自律仕分けの検証', 25)
        text(d, (28,54), f'd1-omni-600M ／ RTX 3070 ／ 公式system_one ／ {speed}倍速再生', 16, '#9eb1cb')
        d.rounded_rectangle((20,95,365,630), radius=14, fill='#162237')
        d.rounded_rectangle((380,95,905,630), radius=14, fill='#162237')
        d.rounded_rectangle((920,95,1260,630), radius=14, fill='#162237')
        active = next((e for e in reversed(case_events) if e['time'] <= t), None)
        infer = next((e for e in reversed(inference_events) if e['time'] <= t and active and e['case_id']==active['case_id']), None)
        arrived = any(e['time'] <= t and active and e['case_id']==active['case_id'] for e in arrivals)
        if active:
            canvas.paste(images[active['image']], (37,145))
            text(d, (37,108), f"入力画像 ／ 検査{active['case_id'].split('_')[-1]}", 21)
            text(d, (37,470), '人工的に生成したワッシャー画像', 16, '#9eb1cb')
            text(d, (37,500), '正解ラベルは評価にだけ使用', 14, '#9eb1cb')
            text(d, (37,530), 'AIには正解を渡していません', 14, '#9eb1cb')
        text(d, (399,108), 'turtlesimの実画面と実測姿勢の再生', 18)
        mono = t-summary['clock_wall_origin']+summary['clock_monotonic_origin']
        desired = max(0, bisect.bisect_right(capture_times,mono)-1)
        while capture.isOpened() and raw_index < desired:
            ok, fr = capture.read()
            if not ok:
                break
            raw_frame, raw_index = fr, raw_index+1
        if raw_frame is not None:
            canvas.paste(Image.fromarray(cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)).resize((240,210)), (395,146))
        frame_age=max(0.,mono-capture_times[min(max(raw_index,0),len(capture_times)-1)])
        text(d, (399,366), f'画面取得からの経過：{frame_age:.2f}秒', 14,
             '#ffc767' if frame_age>.3 else '#9eb1cb')
        # Measured pose trace on a separate labelled coordinate map.
        left, top, side = 645, 147, 240
        d.rectangle((left, top, left+side, top+side), fill='#0e192a', outline='#40516b')

        def point(x,y):
            return left+x/11.1*side, top+side-y/11.1*side

        for label, (x,y) in TARGETS.items():
            px,py = point(x,y)
            d.ellipse((px-5,py-5,px+5,py+5), fill=COLORS[label])
            text(d, (px-25,py+8), LABEL_JA[label], 14, COLORS[label], max_width=70)
        idx = max(0,bisect.bisect_right(pose_times,t)-1)
        trail = [point(p['x'],p['y']) for p in poses[:idx+1:3]]
        if len(trail)>1:
            d.line(trail, fill='#9eaeff', width=2)
        p = poses[idx]
        px,py = point(p['x'],p['y'])
        d.ellipse((px-5,py-5,px+5,py+5), fill='#ffffff')
        text(d, (399,390), f"実測姿勢 x={p['x']:.2f} y={p['y']:.2f} 向き={p['theta']:.2f}", 18)
        text(d, (399,425), '姿勢：/turtle1/pose ／ 速度：/turtle1/cmd_vel', 14, '#9eb1cb')
        text(d, (399,457), '許容誤差0.12 ／ 指令更新の監視0.5秒', 14, '#9eb1cb')
        stage = '目標に到達' if arrived else (LABEL_JA[infer['route']]+'の場所へ移動中' if infer else '画像を検査中 ／ ロボットは停止')
        if p['phase']=='low_confidence_stop':
            stage='確信度が低いため安全停止'
        text(d, (399,510), stage, 21, '#49d6a2' if arrived else '#ffc767')
        text(d, (399,555), f'実動作の経過時間：{t-start:.1f}秒', 18)
        text(d, (938,108), 'AIが出力した判定確率', 21)
        if infer:
            response = infer['response']
            a = response['result']['answers']['inspection']
            for j, label in enumerate(TARGETS):
                y = 164+j*92
                prob = a['probabilities'][label]
                text(d, (940,y), f'{LABEL_JA[label]}（{label}） {prob:.3f}', 21, COLORS[label])
                d.rectangle((940,y+34,1230,y+49), fill='#293952')
                d.rectangle((940,y+34,940+290*prob,y+49), fill=COLORS[label])
            text(d, (938,465), f"GPU推論：{response['inference_seconds']*1000:.0f}ミリ秒", 18)
            text(d, (938,500), f"移動先：{LABEL_JA[infer['route']]}", 21, COLORS[infer['route']])
            text(d, (938,538), '低確信度による再検査' if infer['reason']=='low_confidence' else 'モデルの判定に従う', 16, '#9eb1cb')
        else:
            text(d, (938,170), '実モデルの推論を待っています', 16, '#9eb1cb')
        completed = sum(e['time']<=t for e in arrivals)
        text(d, (28,650), f'目標到達：{completed}/{len(case_events)}件 ／ アームを持たない2次元移動ロボット', 18)
        text(d, (28,685), '確率は未校正のsoftmax値です。画面・姿勢・軌跡は実際のROS動作記録に基づきます。', 14, '#9eb1cb')
        if vt < intro or vt > intro+elapsed/speed:
            d.rectangle((80,170,1200,540), fill='#0d182c', outline='#58708f', width=2)
            if vt < intro:
                lines = [('画像 → AI判定 → 姿勢フィードバック制御', 32),
                         ('仮想検査：合格・不合格・再検査の3か所へ自律移動', 21),
                         ('人工画像9枚：正常・破損・遮蔽を入力', 21),
                         ('NVIDIA GPUでの実推論とROS 2 turtlesimの実動作', 21),
                         ('この導入は静止画です。続いて実測記録を再生します。', 18)]
            else:
                times = [c['inference_seconds'] for c in summary['cases']]
                lines = [('実動作の測定完了', 40),
                         (f"{len(case_events)}/{len(case_events)}件の目標に到達 ／ 移動先は{summary['distinct_routes']}種類", 25),
                         (f"平均GPU推論：{statistics.mean(times)*1000:.0f}ミリ秒 ／ AI正解率：{summary['raw_accuracy']:.1%}", 21),
                         ('誤った合格判定が残っており、実用的な検査精度は未達です。', 18),
                         ('この結果表示は静止画です。追加の動作を示すものではありません。', 18)]
            for j, (line,size) in enumerate(lines):
                text(d, (110,205+j*63), line, size, max_width=1060)
        proc.stdin.write(canvas.tobytes())
    proc.stdin.close()
    if proc.wait()!=0:
        raise RuntimeError('FFmpeg failed')
    capture.release()
    probe = subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_name,width,height',
                                      '-of','json', str(dest)], text=True)
    measured = json.loads(probe)
    assert 60<=float(measured['format']['duration'])<=180
    (dest.parent/'video_metadata.json').write_text(json.dumps({**measured, 'playback_speed':speed,
                     'measured_run_seconds':elapsed, 'intro_seconds':intro, 'outro_seconds':outro,
                     'presentation_language':'ja', 'font':'M PLUS 1p'}, indent=2))
    print(str(dest), probe)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output',type=Path,default=Path('output/submission/inspection_demo.mp4'))
    args=parser.parse_args()
    video(args.run or Path(Path('output/latest_run.txt').read_text().strip()), args.output)
