import cv2
import time

class CameraStream:
    def __init__(self, camera_id, source):
        self.camera_id=str(camera_id)
        self.source=int(source) if str(source).isdigit() else source
        self.capture=cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}: {source}")
        self.fps=self.capture.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_index=0
        self.started=time.time()

    def read(self):
        ok,frame=self.capture.read()
        if not ok:
            # Loop local video files; live streams will simply return no frame.
            if isinstance(self.source,str) and not self.source.lower().startswith("rtsp"):
                self.capture.set(cv2.CAP_PROP_POS_FRAMES,0)
                ok,frame=self.capture.read()
            if not ok: return None
        self.frame_index+=1
        return frame

    def release(self):
        self.capture.release()
