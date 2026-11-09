from ultralytics import YOLO

VEHICLE_CLASS_IDS={2,3,5,7}  # COCO: car, motorcycle, bus, truck

class VehicleDetector:
    def __init__(self, weights="yolov8n.pt", confidence=0.35, device="cpu"):
        self.model=YOLO(weights)
        self.confidence=confidence
        self.device=device

    def detect(self, frame):
        result=self.model.predict(frame,conf=self.confidence,
            classes=list(VEHICLE_CLASS_IDS),device=self.device,verbose=False)[0]
        items=[]
        if result.boxes is None: return items
        for b in result.boxes:
            cid=int(b.cls.item())
            items.append({
                "box":tuple(map(int,b.xyxy[0].tolist())),
                "class":self.model.names.get(cid,str(cid)),
                "confidence":float(b.conf.item())
            })
        return items
