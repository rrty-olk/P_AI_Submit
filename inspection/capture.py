"""Timestamp every actual X11 frame with the same monotonic clock as ROS.

FFmpeg's X11 wall-clock PTS can drift/step in WSL, so they are not used to
synchronize the final dashboard. The untouched frame sequence is retained.
"""
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import cv2
import numpy as np


def main():
    run=Path(sys.argv[1])
    grab=subprocess.Popen(['ffmpeg','-nostdin','-loglevel','error','-f','x11grab','-video_size','640x560',
                           '-framerate','10','-i',':99','-f','rawvideo','-pix_fmt','bgr24','-'],
                          stdout=subprocess.PIPE)
    writer=cv2.VideoWriter(str(run/'turtlesim_raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(640,560))
    if not writer.isOpened():
        raise RuntimeError('Cannot create raw screen recording')
    running=True
    def stop(*_):
        nonlocal running
        running=False
    signal.signal(signal.SIGINT,stop)
    signal.signal(signal.SIGTERM,stop)
    count=0
    try:
        with open(run/'capture_timestamps.jsonl','w',buffering=1) as log:
            while running:
                data=grab.stdout.read(640*560*3)
                stamp=time.monotonic()
                if len(data)!=640*560*3:
                    raise RuntimeError('X11 frame stream ended prematurely')
                writer.write(np.frombuffer(data,dtype=np.uint8).reshape(560,640,3))
                log.write(json.dumps({'frame':count,'monotonic':stamp})+'\n')
                count+=1
    finally:
        grab.terminate()
        grab.wait(timeout=10)
        writer.release()


if __name__=='__main__':
    main()
