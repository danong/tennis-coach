#!/usr/bin/env python3
"""Render clean start/release/peak/impact frames from smoothed metadata."""
import argparse, json, subprocess, tempfile
from pathlib import Path
import cv2, pandas as pd
from find_impact import choose_frame

def main():
 p=argparse.ArgumentParser();p.add_argument('csv',type=Path);p.add_argument('video',type=Path);p.add_argument('stages',type=Path);p.add_argument('--prefix',required=True);a=p.parse_args()
 d=pd.read_csv(a.csv); stages=json.loads(a.stages.read_text())['stages']
 probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=r_frame_rate','-of','csv=p=0',str(a.video)],capture_output=True,text=True,check=True).stdout.strip().rstrip(',')
 n,den=map(float,probe.split('/')); fps=n/den
 frames={k:round(v*fps) for k,v in stages.items()}
 release=frames['release']; impact=choose_frame(d)[0][1]
 peak=int(d.iloc[release:impact]['SmoothY'].idxmin())
 frames.update(peak=peak,impact=impact)
 order={'start':'00-start','release':'01-release','peak':'02-peak','impact':'03-impact'}
 for name,frame in frames.items():
  if name not in order: continue
  with tempfile.NamedTemporaryFile(suffix='.png') as t:
   subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-ss',str(frame/fps),'-i',str(a.video),'-frames:v','1',t.name],check=True)
   image=cv2.imread(t.name)
  out=Path(f'{a.prefix}-{order[name]}.png'); cv2.imwrite(str(out),image); print(f'{name}: frame {frame} -> {out}')
if __name__=='__main__':main()
