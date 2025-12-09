import sys
sys.path.append('/app')

import torch, json
import torchvision.transforms as T
import cv2
import os
from training.model_builder import create_ssd_model
from sort import Sort
import numpy as np

# ========================
# CONFIGURAZIONE
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../saved_models/best_model.pth")  # percorso al modello
VIDEO_PATH = os.path.join(BASE_DIR, "../inference/video4.mp4")
CONF_THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "data_dir": os.path.join(BASE_DIR, "../training/COCO-Home-Objects"),
    "annotations_file": "{}/_annotations.coco.json",
}

# ========================
# CARICAMENTO CLASSI
# ========================
'''data_dir = config['data_dir']
annotations_file_template = config['annotations_file']
train_ann_file = os.path.join(data_dir, annotations_file_template.format('train'))
train_img_dir = os.path.join(data_dir, 'train')

temp_dataset = CocoHomeDataset(images_dir=train_img_dir, annotations_file=train_ann_file)
category_id_to_name = {cat['id']: cat['name'] for cat in temp_dataset.coco.loadCats(temp_dataset.category_ids)}
HOME_CLASSES = ["__background__"] + [category_id_to_name[cid] for cid in temp_dataset.category_ids]'''

# To Create Annotations File
'''json_string = json.dumps(HOME_CLASSES, indent=4)
with open("home_classes.json", "w") as f:
    f.write(json_string)'''

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
    #T.Resize((300, 300)),
    T.ToTensor(),
])

# ========================
# INFERENZA SU VIDEO
# ========================

# Inizializza SORT tracker
tracker = Sort(max_age=20, min_hits=1, iou_threshold=0.2)
# max_age è il numero massimo di frame la cui entità viene seguita
# min_hits è il numero di predizioni sulla stessa entità necessarie per avere un ID
# iou_threshold di quanto si può spostare la box della predizione

cap = cv2.VideoCapture(VIDEO_PATH)

frame_count = 0
DETECT_EVERY_N_FRAMES = 20

if not cap.isOpened():
    raise RuntimeError(f"Impossibile aprire il video {VIDEO_PATH}")

# Funzione per assegnare univocamente un id ad una classe predetta
def compute_iou(box1, box2):
    """Calcola Intersection over Union tra due bbox"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    if box1_area + box2_area == 0:
        return 0.0

    return inter_area / (box1_area + box2_area - inter_area)

track_class_map = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Detection solo ogni N frame
    if frame_count % DETECT_EVERY_N_FRAMES == 0:
        input_tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(input_tensor)[0]

        boxes = outputs['boxes'].cpu().numpy()
        scores = outputs['scores'].cpu().numpy()
        labels = outputs['labels'].cpu().numpy()

        # Filtra detection con confidence alta
        mask = scores >= CONF_THRESHOLD
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        # Prepara detection per SORT (formato: [x1, y1, x2, y2, score])
        if len(boxes) > 0:
            detections = np.column_stack([boxes, scores])
        else:
            detections = np.empty((0, 5))

        # SORT aggiorna i tracker (velocissimo!)
        tracked_objects = tracker.update(detections)

        # Associa ogni detection al suo track ID per INDICE
        for track_obj in tracked_objects:
            track_box = track_obj[:4]
            track_id = int(track_obj[4])

            best_iou = 0
            best_label_idx = -1

            for det_idx, det_box in enumerate(boxes):
                iou_score = compute_iou(track_box, det_box)
                if iou_score > best_iou and iou_score > 0.3:
                    best_iou = iou_score
                    best_label_idx = det_idx

            if best_label_idx != -1:
                track_class_map[track_id] = HOME_CLASSES[labels[best_label_idx]]

        # Disegna gli oggetti tracciati
        for obj in tracked_objects:
            obj_id = int(obj[4])
            class_name = track_class_map.get(obj_id, "unknown")

            x1, y1, x2, y2 = map(int, obj[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{obj_id} {class_name}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("SSD300 Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
