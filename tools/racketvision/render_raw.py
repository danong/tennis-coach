#!/usr/bin/env python3
"""Render raw tracking metadata on the original portrait video."""
import argparse
from pathlib import Path
import cv2
import pandas as pd

EDGES=((11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28),(27,29),(29,31),(28,30),(30,32))
RACKET=("Top","Bottom","Handle","Left","Right")

p=lambda r,n:(round(r[f"{n}X"]),round(r[f"{n}Y"]))
def main():
 a=argparse.ArgumentParser();a.add_argument("video",type=Path);a.add_argument("csv",type=Path);a.add_argument("--output",type=Path,required=True);x=a.parse_args();d=pd.read_csv(x.csv);c=cv2.VideoCapture(str(x.video));fps=c.get(cv2.CAP_PROP_FPS) or 30;w,h=1080,1920;out=cv2.VideoWriter(str(x.output),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
 for i,(_,r) in enumerate(d.iterrows()):
  ok,f=c.read()
  if not ok: raise RuntimeError(f"video ended at frame {i}")
  f=cv2.rotate(f,cv2.ROTATE_90_CLOCKWISE); o=f.copy()
  if r.get("Visibility",0)>0: cv2.circle(o,(round(r.X),round(r.Y)),14,(0,255,255),3)
  pose=[r.get(f"Pose{i}X") for i in range(33)]
  for u,v in EDGES:
   if pd.notna(pose[u]) and pd.notna(pose[v]): cv2.line(o,(round(pose[u]),round(r[f"Pose{u}Y"])),(round(pose[v]),round(r[f"Pose{v}Y"])),(0,165,255),2)
  for n in RACKET:
   if pd.notna(r.get(f"{n}X")): cv2.circle(o,p(r,n),7,(255,0,255),-1)
  out.write(o)
 c.release();out.release();print(f"Wrote {x.output}")
if __name__=="__main__":main()
