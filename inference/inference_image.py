import torch, json
import torchvision.transforms as T
import cv2
import os
from training.dataset_builder import CocoHomeDataset
from training.model_builder import create_ssd_model


# ========================
# CONFIGURAZIONE
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = "../saved_models/best_model.pth"  # percorso al modello
CONF_THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")

config = {
    "data_dir": "../training/COCO-Home-Objects",
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
    T.Resize((300, 300)),
    T.ToTensor(),
])

# ========================
# FUNZIONE DI INFERENZA
# ========================
def infer_image(image_path):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    scale_x = w / 300
    scale_y = h / 300
    if image is None:
        print(f"[ERRORE] Immagine non trovata: {image_path}")
        return

    orig = image.copy()
    input_tensor = transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)[0]

    boxes = outputs['boxes'].cpu().numpy()
    scores = outputs['scores'].cpu().numpy()
    labels = outputs['labels'].cpu().numpy()

    for box, score, label in zip(boxes, scores, labels):
        if score >= CONF_THRESHOLD:
            x1, y1, x2, y2 = map(int, box)
            x1 = int(x1 * scale_x)
            x2 = int(x2 * scale_x)
            y1 = int(y1 * scale_y)
            y2 = int(y2 * scale_y)
            class_name = HOME_CLASSES[label]
            cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(orig, f"{class_name} {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("SSD300 Detection - Image", orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# ========================
# ESEMPIO DI USO
# ========================
#test_image = os.path.join(train_img_dir, "000123.jpg")  # cambia immagine
test_image = os.path.join("foto.jpg")  # cambia immagine
infer_image(test_image)
