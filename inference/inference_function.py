import sys
sys.path.append('/app')

import torch, json, time
import torchvision.transforms as T
import cv2
import os
from training.model_builder import create_ssd_model

class InferenceFunction:
    def __init__(self):
        # ========================
        # CONFIGURAZIONE
        # ========================
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(BASE_DIR, "../saved_models/best_model.pth")  # percorso al modello
        self.CONF_THRESHOLD = 0.5
        self.DEVICE = torch.device("cpu" if torch.mps.is_available() else "cpu")

        # ========================
        # CARICAMENTO CLASSI
        # ========================

        # Load Annotations File Created
        with open("../inference/home_classes.json", "r") as f:
            self.HOME_CLASSES = json.load(f)

        num_classes = len(self.HOME_CLASSES)

        print(f"[INFO] {num_classes} Classi caricate: {self.HOME_CLASSES}")

        # ========================
        # CARICAMENTO MODELLO
        # ========================
        self.model = create_ssd_model(num_classes=num_classes, device=self.DEVICE)
        self.model.head.classification_head.num_classes = num_classes
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.DEVICE))
        self.model.to(self.DEVICE)
        self.model.eval()

        # ========================
        # TRASFORMAZIONE
        # ========================
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((300, 300)),
            T.ToTensor(),
        ])


    def predict(self, frame):
        height, width = frame.shape[:2]

        scale_x = width / 300
        scale_y = height / 300

        fps_start_time = time.time()
        processed_frames = 0

        t0 = time.time()
        input_tensor = self.transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.DEVICE)

        with torch.no_grad():
            outputs = self.model(input_tensor)[0]

        infer_time = (time.time() - t0) * 1000.0  # ms

        boxes = outputs['boxes'].cpu().numpy()
        scores = outputs['scores'].cpu().numpy()
        labels = outputs['labels'].cpu().numpy()

        # Filtra detection con confidence alta
        mask = scores >= self.CONF_THRESHOLD
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        # FPS globale
        processed_frames += 1
        elapsed = time.time() - fps_start_time
        fps = processed_frames / elapsed if elapsed > 0 else 0.0

        # ------------------
        # COSTRUZIONE JSON
        # ------------------
        frame_annotations = {
            "inference_time_ms": infer_time,
            "fps": fps,
            "objects": []
        }

        # Disegna le detection direttamente (senza tracking)
        for box, score, label in zip(boxes, scores, labels):
            # box è nel sistema 300x300, riscalo alle dimensioni del frame originale
            x1 = int(box[0] * scale_x)
            y1 = int(box[1] * scale_y)
            x2 = int(box[2] * scale_x)
            y2 = int(box[3] * scale_y)

            class_name = self.HOME_CLASSES[int(label)]

            frame_annotations["objects"].append({
                "class_name": class_name,
                "score": float(score),
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }
            })

        return frame_annotations



