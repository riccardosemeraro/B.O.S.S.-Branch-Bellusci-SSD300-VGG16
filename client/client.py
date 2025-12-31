import sys, os
sys.path.append('/app')

import time
import json
import cv2
import paho.mqtt.client as mqtt
from broker.configuration import *
from gui.gui_v5 import SmartGlassesGUI

from pathlib import Path
SCRIPT_DIR = Path(__file__).parent.parent

# --------
# FUNZIONI
# --------

# Disegno i BOXES ricevuti dal server
def draw_boxes_from_json(frame, data):

    # Ciclo su ogni oggetto del JSON
    for obj in data["objects"]:
        # Estraggo informazioni dal JSON
        bbox = obj["bbox"]
        class_name = obj["class_name"]
        score = obj["score"]

        # Estraggo coordinate x1-y1 e altezza-larghezza del bbox
        x1 = int(bbox["x1"])
        y1 = int(bbox["y1"])
        x2 = int(bbox["x2"])
        y2 = int(bbox["y2"])

        # Disegno il rettangolo
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Scrivo la classe sopra il rettangolo
        cv2.putText(
            frame,
            f"{class_name} {score:.2f}",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    return frame

# Funzione eseguita quando si connette a MQTT
def on_connect(client, userdata, flags, rc, properties):
    print("Connesso al broker con codice", rc)
    client.subscribe(TOPIC_PRED)

# Funzione eseguita quando riceve un messaggio da MQTT
def on_message(client, userdata, msg):
    global JSON_BOXES
    print(f"Ricevuto da: {msg.topic}: {msg.payload.decode()}")
    JSON_BOXES = msg.payload.decode()

# -----------
# MQTT CLIENT
# -----------

client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message

# ------------------
# VARIABILI GLOBALI
# ------------------

JSON_BOXES = None

# --------------------
# INIZIALIZZAZIONE GUI
# --------------------

cv2.namedWindow("SmartGlasses GUI")
smartGUI = SmartGlassesGUI()

# --------------------------
# CONNESSIONE AL BROKER MQTT
# --------------------------

client.connect(BROKER_CONTAINER, BROKER_PORT, keepalive=60)
print("Mi sto collegando al broker MQTT... 😃")
client.loop_start()
# loop_forever impediva di avere il ruolo di publisher/subscriber,
# dunque loop_start e loop_stop per gestire il flusso

# ----
# MAIN
# ----

VIDEO_PATH = SCRIPT_DIR / "inference/video4.mp4"

# Acquisisco il video
cap = cv2.VideoCapture(str(VIDEO_PATH))

# Inizializzo variabili per il count dei frame da inviare
frame_count = 0
DETECT_EVERY_N_FRAMES = 15

# Gestisco eventuali errori
if not cap.isOpened():
    raise RuntimeError(f"Impossibile aprire il video {VIDEO_PATH}")

while True:
    # leggo il primo frame
    ret, frame = cap.read()

    # Riavvia video dall'inizio alla sua conclusione
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Aggiorno variabile count
    frame_count += 1

    # Conversione frame cv2 in bytes da mandare tramite protocollo MQTT
    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    frame_to_bytes = buffer.tobytes()

    # Condizione di efficienza, ogni DETECT_EVERY_N_FRAMES aggiorna JSON_BOXES mandando un nuovo frame da inferire
    if frame_count % DETECT_EVERY_N_FRAMES == 0:
        client.publish(TOPIC_FRAME, payload=frame_to_bytes, qos=1, retain=False)

    # Aggiorno Boxes
    if JSON_BOXES is not None:
        # Converte in JSON il JSON_BOXES ricevuto
        data = json.loads(JSON_BOXES)
        # Aggiorno Inference Time sulla GUI
        inf_ms = data.get("inference_time_ms", None)
        smartGUI.setInferenceTime(inf_ms)

        # Disegno bbox sul frame
        frame = draw_boxes_from_json(frame, data)

    # Aggiorno GUI
    canvas = smartGUI.update_canvas(frame)

    # Emulazione FPS reali di una webcam standard a 30 FPS, limitando l'analisi a 1 frame ogni 0.01 secondi
    time.sleep(0.01)

    # Mostro il risultato
    cv2.imshow("SmartGlasses GUI", canvas)

    # Tasto Q per terminare il client
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascio e distruzione GUI
cap.release()
cv2.destroyAllWindows()

# --------------------
# CHIUSURA CONNESSIONE
# --------------------
client.loop_stop()
