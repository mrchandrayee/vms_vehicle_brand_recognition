from pathlib import Path
import torch
from PIL import Image
from torchvision import models, transforms

class EfficientNetBrandClassifier:
    """Loads a user-trained EfficientNet-B0 checkpoint; no brand weights are bundled."""
    def __init__(self, checkpoint, classes, device="cpu", confidence=0.55):
        self.path=Path(checkpoint)
        if not self.path.is_file():
            raise FileNotFoundError(f"Brand checkpoint not found: {self.path}")
        self.device=torch.device(device if device=="cpu" or torch.cuda.is_available() else "cpu")
        payload=torch.load(self.path,map_location=self.device)
        if isinstance(payload,dict) and "state_dict" in payload:
            state=payload["state_dict"]
            classes=payload.get("classes",classes)
        else:
            state=payload
        self.classes=list(classes)
        if not self.classes:
            raise ValueError("At least one brand class is required")
        self.net=models.efficientnet_b0(weights=None)
        in_features=self.net.classifier[1].in_features
        self.net.classifier[1]=torch.nn.Linear(in_features,len(self.classes))
        # Accept checkpoints saved from DataParallel.
        state={k.removeprefix("module."):v for k,v in state.items()}
        self.net.load_state_dict(state,strict=True)
        self.net.to(self.device).eval()
        self.confidence=float(confidence)
        self.preprocess=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])

    @torch.inference_mode()
    def predict(self, bgr_crop):
        if bgr_crop is None or bgr_crop.size==0:
            return None,0.0
        import cv2
        rgb=cv2.cvtColor(bgr_crop,cv2.COLOR_BGR2RGB)
        tensor=self.preprocess(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        probs=torch.softmax(self.net(tensor),dim=1)[0]
        score,index=torch.max(probs,dim=0)
        score=float(score.item())
        if score<self.confidence:
            return None,score
        return self.classes[int(index.item())],score
