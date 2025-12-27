import sys
import time

sys.path.append('/app')

import json
import cv2
import os
import paho.mqtt.client as mqtt
from broker.configuration import *
from gui.gui_v5 import SmartGlassesGUI

# ------------------
# FUNZIONI
# ------------------

def draw_boxes_from_json(frame, data):

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

def on_connect(client, userdata, flags, rc, properties):
    print("Connesso al broker con codice", rc)
    client.subscribe(TOPIC_PRED)


def on_message(client, userdata, msg):
    global JSON_BOXES
    print(f"Ricevuto da: {msg.topic}: {msg.payload.decode()}")
    JSON_BOXES = msg.payload.decode()

# ------------------
# MQTT CLIENT
# ------------------

client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message

# ------------------
# VARIABILI GLOBALI
# ------------------

JSON_BOXES = None
smartGUI = SmartGlassesGUI()

# ------------------
# CONNESSIONE AL BROKER MQTT
# ------------------

client.connect(BROKER_CONTAINER, BROKER_PORT, keepalive=60)
print("Mi sto collegando al broker MQTT... 😃")
client.loop_start()
# loop_forever impediva di avere il ruolo di publisher/subscriber,
# dunque loop_start e loop_stop per gestire il flusso

# ------------------
# MAIN
# ------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "../inference/video4.mp4")

#cap = cv2.VideoCapture(VIDEO_PATH)
cap = cv2.VideoCapture(1)

frame_count = 0
DETECT_EVERY_N_FRAMES = 15

if not cap.isOpened():
    raise RuntimeError(f"Impossibile aprire il video {VIDEO_PATH}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    frame_to_bytes = buffer.tobytes()

    if frame_count % DETECT_EVERY_N_FRAMES == 0:
        client.publish(TOPIC_FRAME, payload=frame_to_bytes, qos=1, retain=False)

    if JSON_BOXES is not None:
        data = json.loads(JSON_BOXES)
        # opzionale: mostrare inference time e fps sul frame
        inf_ms = data.get("inference_time_ms", None)
        smartGUI.setInferenceTime(inf_ms)

        frame = draw_boxes_from_json(frame, data)

    canvas = smartGUI.update_canvas(frame)
    time.sleep(0.01)

    cv2.imshow("SmartGlasses GUI", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ------------------
# CHIUSURA CONNESSIONE
# ------------------
client.loop_stop()
