import io
from typing import List, Dict, Any

import litserve as ls
from fastapi import UploadFile
from PIL import Image
from ultralytics import YOLO

# TRUCO TEMPORAL: Usamos el modelo base para probar la API hoy.
# Cuando termine Kaggle, cambiarás esto a "best.pt"
MODEL_PATH = "best.onnx"

class LogisticsAPI(ls.LitAPI):
    def setup(self, device: str):
        self.model = YOLO(MODEL_PATH)

    def decode_request(self, request: Dict[str, Any]):
        if isinstance(request, dict) and "image" in request:
            f: UploadFile = request["image"]
            image = Image.open(io.BytesIO(f.file.read())).convert("RGB")
            return image
        raise ValueError("Expected form-data with 'image'")

    def predict(self, image: Image.Image) -> List[Dict[str, Any]]:
        results = self.model.predict(image, conf=0.25, imgsz=640, verbose=False)[0]
        dets = []
        for b in results.boxes:
            cls_id = int(b.cls[0].item())
            dets.append({
                "class_id": cls_id,
                "class_name": results.names[cls_id],
                "confidence": float(b.conf[0].item()),
                "xyxy": [float(x) for x in b.xyxy[0].tolist()],
            })
        return dets

    def encode_response(self, output: List[Dict[str, Any]]):
        return {"detections": output}

if __name__ == "__main__":
    server = ls.LitServer(LogisticsAPI(), accelerator="auto")
    server.run(port=8000)