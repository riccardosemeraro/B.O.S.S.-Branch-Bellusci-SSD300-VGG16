import sys
sys.path.append("/Users/riccardosemeraro/Downloads/B.O.S.S.-Branch-Bellusci-SSD300-VGG16/training")

import torch
import torchvision.transforms as T
import cv2
import os
from dataset_builder import CocoHomeDataset
from model_builder import create_ssd_model

# ========================
# CONFIGURAZIONE
# ========================
MODEL_PATH = "../saved_models/best_model.pth"  # percorso al modello
CONF_THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps") #cpu

config = {
    "data_dir": "../training/COCO-Home-Objects",
    "annotations_file": "{}/_annotations.coco.json",
}

# ========================
# CARICAMENTO CLASSI
# ========================
data_dir = config['data_dir']
annotations_file_template = config['annotations_file']
train_ann_file = os.path.join(data_dir, annotations_file_template.format('train'))
train_img_dir = os.path.join(data_dir, 'train')

temp_dataset = CocoHomeDataset(images_dir=train_img_dir, annotations_file=train_ann_file)
category_id_to_name = {cat['id']: cat['name'] for cat in temp_dataset.coco.loadCats(temp_dataset.category_ids)}
HOME_CLASSES = ["__background__"] + [category_id_to_name[cid] for cid in temp_dataset.category_ids]

print(f"[INFO] Classi caricate: {HOME_CLASSES}")

# ========================
# CARICAMENTO MODELLO 
# ========================
num_classes = temp_dataset.num_classes
model = create_ssd_model(num_classes=num_classes, device=DEVICE)
model.head.classification_head.num_classes = temp_dataset.num_classes
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
# FUNZIONE INFERENZA WEBCAM
# ========================
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    raise RuntimeError("Impossibile aprire la webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(input_tensor)[0]

    boxes = outputs['boxes'].cpu().numpy()
    scores = outputs['scores'].cpu().numpy()
    labels = outputs['labels'].cpu().numpy()

    for box, score, label in zip(boxes, scores, labels):
        if score >= CONF_THRESHOLD:
            x1, y1, x2, y2 = map(int, box)
            class_name = HOME_CLASSES[label]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("SSD300 Detection - Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
