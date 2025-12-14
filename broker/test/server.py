import time
import paho.mqtt.client as mqtt
from broker.configuration import *

def on_connect(client, userdata, flags, rc):
    print("Connesso al broker con codice", rc)
    client.subscribe(TOPIC_FRAME)


def on_message(client, userdata, msg):
    print(f"Messaggio su {msg.topic}: {msg.payload.decode()}")

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER_CONTAINER, BROKER_PORT, keepalive=60)
client.loop_start()
# loop_forever impediva di avere il ruolo di publisher/subscriber,
# dunque loop_start e loop_stop per gestire il flusso

for i in range(5):
    payload = f"ciao {i}"
    print("Pubblico:", payload)
    client.publish(TOPIC_PRED, payload=payload, qos=0, retain=False)
    time.sleep(1)
