from broker.configuration import *
import paho.mqtt.client as mqtt
import sys
sys.path.append('/app')

import torch, json, time
import numpy as np
import cv2

# ========================
# IMPORT SCRIPT INFERENZA
# ========================
from inference.inference_function import InferenceFunction
inference_function = InferenceFunction()

def on_connect(client, userdata, flags, rc, properties):
    print("Connesso al broker con codice", rc)
    client.subscribe(TOPIC_FRAME)


def on_message(client, userdata, msg):
    nparr = np.frombuffer(msg.payload, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Esegue l'inferenza sul frame ricevuto
    frame_annotations = inference_function.predict(frame)

    # Pubblica le annotazioni sul topic delle predizioni
    client.publish(TOPIC_PRED, payload=json.dumps(frame_annotations), qos=2, retain=False)

client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER_CONTAINER, BROKER_PORT, keepalive=60)
client.loop_forever()