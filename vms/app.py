import argparse, csv, json, time
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from .camera import CameraStream
from .detector import VehicleDetector
from .tracker import CentroidTracker
from .brand import EfficientNetBrandClassifier

def parse_args():
    p=argparse.ArgumentParser(description="Multi-camera VMS vehicle/brand recognition")
    p.add_argument("--config",default="cameras.json")
    p.add_argument("--output",default="runs/vms")
    p.add_argument("--no-brand",action="store_true",
                   help="Run vehicle detection/tracking without brand classifier")
    return p.parse_args()

def resize_tile(frame,width,height):
    return cv2.resize(frame,(width,height),interpolation=cv2.INTER_AREA)

def main():
    args=parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding="utf-8"))
    streams=[]
    for item in cfg["cameras"]:
        streams.append(CameraStream(item["id"],item["source"]))
    if not streams:
        raise RuntimeError("No cameras configured in cameras.json")
    detector=VehicleDetector(cfg.get("vehicle_model","yolov8n.pt"),
        cfg.get("vehicle_confidence",0.35),cfg.get("device","cpu"))
    classifier=None
    checkpoint=cfg.get("brand_checkpoint","models/vehicle_brand_efficientnet_b0.pth")
    if not args.no_brand and Path(checkpoint).is_file():
        classifier=EfficientNetBrandClassifier(checkpoint,cfg.get("brand_classes",[]),
            cfg.get("device","cpu"),cfg.get("brand_confidence",0.55))
        print("[brand] EfficientNet-B0 checkpoint loaded")
    else:
        print("[brand] No checkpoint found; showing vehicle class only.")
    trackers={s.camera_id:CentroidTracker(vote_count=cfg.get("brand_vote_count",3))
              for s in streams}
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    log_path=output/"vehicle_events.csv"
    logged=set()
    cols=max(1,int(cfg.get("grid_columns",2)))
    tile_w,tile_h=640,360
    try:
        with log_path.open("w",newline="",encoding="utf-8") as f:
            log=csv.writer(f)
            log.writerow(["timestamp","camera_id","track_id","vehicle_type",
                          "brand","brand_confidence","x1","y1","x2","y2"])
            while True:
                tiles=[]
                for stream in streams:
                    frame=stream.read()
                    if frame is None:
                        tile=np.zeros((tile_h,tile_w,3),dtype=np.uint8)
                        cv2.putText(tile,f'{stream.camera_id} - NO SIGNAL',(25,60),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
                        tiles.append(tile); continue
                    now=time.time()-stream.started
                    detections=detector.detect(frame)
                    tracker=trackers[stream.camera_id]
                    tracks=tracker.update([d["box"] for d in detections],now)
                    for tr in tracks:
                        # Match track to nearest detection by center.
                        match=None; best=1e9
                        for d in detections:
                            b=d["box"]; cx=(b[0]+b[2])//2; cy=(b[1]+b[3])//2
                            dist=((cx-tr.center[0])**2+(cy-tr.center[1])**2)**0.5
                            if dist<best: best=dist; match=d
                        if match is None: continue
                        x1,y1,x2,y2=tr.box
                        brand="unclassified"
                        if classifier and not tr.locked:
                            # Slightly expanded vehicle ROI; resize/crop logic lives in classifier.
                            pad_x=int((x2-x1)*0.04); pad_y=int((y2-y1)*0.04)
                            crop=frame[max(0,y1-pad_y):min(frame.shape[0],y2+pad_y),
                                       max(0,x1-pad_x):min(frame.shape[1],x2+pad_x)]
                            name,score=classifier.predict(crop)
                            if name:
                                tracker.add_brand_vote(tr,name,score)
                        brand=tr.brand if tr.locked else "brand pending"
                        color=(0,220,0) if tr.locked else (0,165,255)
                        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                        label=f'ID {tr.track_id} | {match["class"]} | {brand}'
                        cv2.putText(frame,label,(x1,max(18,y1-7)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.48,color,2)
                        if tr.locked:
                            key=(stream.camera_id,tr.track_id)
                            if key not in logged:
                                log.writerow([time.strftime("%Y-%m-%d %H:%M:%S"),
                                    stream.camera_id,tr.track_id,match["class"],
                                    tr.brand,round(tr.brand_score,4),x1,y1,x2,y2])
                                f.flush(); logged.add(key)
                    cv2.rectangle(frame,(0,0),(190,32),(15,15,15),-1)
                    cv2.putText(frame,f'VMS AI | {stream.camera_id}',(8,22),
                        cv2.FONT_HERSHEY_SIMPLEX,0.58,(0,255,0),2)
                    if cfg.get("show_fps",True):
                        cv2.putText(frame,f'Objects: {len(tracks)}',(8,frame.shape[0]-12),
                            cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1)
                    tiles.append(resize_tile(frame,tile_w,tile_h))
                rows=[]
                for start in range(0,len(tiles),cols):
                    row=tiles[start:start+cols]
                    while len(row)<cols:
                        row.append(np.zeros((tile_h,tile_w,3),dtype=np.uint8))
                    rows.append(np.hstack(row))
                grid=np.vstack(rows)
                cv2.imshow("VMS AI - Multi Camera Vehicle Brand Recognition",grid)
                key=cv2.waitKey(1)&0xFF
                if key==ord("q"): break
                if key==ord("r"):
                    trackers={s.camera_id:CentroidTracker(
                        vote_count=cfg.get("brand_vote_count",3)) for s in streams}
                    logged.clear()
    finally:
        for s in streams: s.release()
        cv2.destroyAllWindows()
    print("Event log:",log_path)
