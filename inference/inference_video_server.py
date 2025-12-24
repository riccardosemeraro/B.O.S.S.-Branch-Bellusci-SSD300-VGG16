import sys
sys.path.append('/app')

import torch, json, time
import torchvision.transforms as T
import cv2
import os
from training.model_builder import create_ssd_model
import numpy as np

# ========================
# CONFIGURAZIONE
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../saved_models/best_model.pth")  # percorso al modello
VIDEO_PATH = os.path.join(BASE_DIR, "../inference/video4.mp4")
CONF_THRESHOLD = 0.7
DEVICE = torch.device("cpu" if torch.mps.is_available() else "cpu")

config = {
    "data_dir": os.path.join(BASE_DIR, "../training/COCO-Home-Objects"),
    "annotations_file": "{}/_annotations.coco.json",
}

# ========================
# CARICAMENTO CLASSI
# ========================

# Load Annotations File Created
with open("home_classes.json", "r") as f:
    HOME_CLASSES = json.load(f)

num_classes = len(HOME_CLASSES)

print(f"[INFO] {num_classes} Classi caricate: {HOME_CLASSES}")

# ========================
# CARICAMENTO MODELLO
# ========================
model = create_ssd_model(num_classes=num_classes, device=DEVICE)
model.head.classification_head.num_classes = num_classes
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ========================
# TRASFORMAZIONE
# ========================
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((300, 300)),
    T.ToTensor(),
])

# ========================
# INFERENZA SU VIDEO
# ========================

def draw_boxes(frame, boxes, scores, labels, infer_time, fps):
    # overlay nero per testo
    cv2.rectangle(frame, (0, 0), (300, 80), (0, 0, 0), -1)

    # Scrivo il tempo di inference
    cv2.putText(frame, f"Inference: {infer_time:.1f} ms", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

    # Scrivo gli FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

    # Disegna le detection direttamente (senza tracking)
    for box, score, label in zip(boxes, scores, labels):
        # box è nel sistema 300x300, riscalo alle dimensioni del frame originale
        x1 = int(box[0] * scale_x)
        y1 = int(box[1] * scale_y)
        x2 = int(box[2] * scale_x)
        y2 = int(box[3] * scale_y)

        class_name = HOME_CLASSES[int(label)]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{class_name} {score:.2f}",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return frame

def draw_boxes_from_json(frame, json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    # opzionale: mostrare inference time e fps sul frame
    inf_ms = data.get("inference_time_ms", None)
    fps = data.get("fps", None)

    cv2.rectangle(frame, (0, 0), (300, 80), (0, 0, 0), -1)

    if inf_ms is not None:
        cv2.putText(frame, f"Inference: {inf_ms:.1f} ms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

    for obj in data["objects"]:
        bbox = obj["bbox"]
        class_name = obj["class_name"]
        score = obj["score"]

        x1 = int(bbox["x1"])
        y1 = int(bbox["y1"])
        x2 = int(bbox["x2"])
        y2 = int(bbox["y2"])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{class_name} {score:.2f}",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return frame


cap = cv2.VideoCapture(VIDEO_PATH)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

scale_x = width / 300
scale_y = height / 300

frame_count = 0
DETECT_EVERY_N_FRAMES = 15

if not cap.isOpened():
    raise RuntimeError(f"Impossibile aprire il video {VIDEO_PATH}")

fps_start_time = time.time()
processed_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Detection solo ogni N frame
    if frame_count % DETECT_EVERY_N_FRAMES == 0:
        t0 = time.time()
        input_tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)[0]

        infer_time = (time.time() - t0) * 1000.0  # ms

        boxes = outputs['boxes'].cpu().numpy()
        scores = outputs['scores'].cpu().numpy()
        labels = outputs['labels'].cpu().numpy()

        # Filtra detection con confidence alta
        mask = scores >= CONF_THRESHOLD
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        # FPS globale (come stavi già facendo)
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

            class_name = HOME_CLASSES[int(label)]

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

        frame = draw_boxes(frame, boxes, scores, labels, infer_time, fps)

        # salva JSON
        json_path = os.path.join(BASE_DIR, f"frame.json")
        with open(json_path, "w") as jf:
            json.dump(frame_annotations, jf, indent=2)

    else:
        x = 2
        # Invia stesso file creato nel caso then

    # Mostra comunque l'ultimo frame aggiornato
    cv2.imshow("Detections", draw_boxes_from_json(frame, "./frame.json"))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()
