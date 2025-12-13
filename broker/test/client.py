from broker.configuration import *
import paho.mqtt.client as mqtt

def on_connect(client, userdata, flags, rc):
    print("Connesso al broker con codice", rc)
    client.subscribe(TOPIC_PRED)

def on_message(client, userdata, msg):
    print(f"Messaggio su {msg.topic}: {msg.payload.decode()}")
    payload = f"Ho Ricevuto il Messaggio!!! 😃"
    print("Pubblico:", payload)
    client.publish(TOPIC_FRAME, payload=payload, qos=0, retain=False)

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER_CONTAINER, BROKER_PORT, keepalive=60)
client.loop_forever()