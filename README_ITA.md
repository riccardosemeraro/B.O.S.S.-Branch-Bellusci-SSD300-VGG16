

# 🚀 B.O.S.S. (Blind Oriented Suggested System) - SSD300 VGG16 Branch

  

[![Python](https://img.shields.io/badge/Python-3.13-FFE873.svg?logo=python&logoColor=blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-orange.svg?logo=pytorch)](https://pytorch.org/) 
[![OPENCV](https://img.shields.io/badge/OpenCV-4.12.0.88-brown.svg?logo=opencv)](https://opencv.org/)

[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg?logo=docker)](https://www.docker.com/) 
[![MQTT](https://img.shields.io/badge/MQTT-Mosquitto-199037.svg?logo=mqtt)](https://mosquitto.org/)
[![COCO](https://img.shields.io/badge/COCO-Home%20Object-purple.svg?logo=roboflow)](https://universe.roboflow.com/objectdetection-uzld5/coco-home-objects)


Questo progetto implementa una rete neurale **SSD300** basata su **VGG16** per il rilevamento di oggetti, integrata in un'architettura publisher-subscriber distribuita con comunicazione MQTT, coadiuvata da una GUI minimale su SmartGlasses Emulati. Obiettivo ultimo è assistere alla navigazioni utenti con limitazioni visive.
  

---

  

## 📋 Tabella dei Contenuti

- [🎯 Obiettivo](#-obiettivo)

- [🛠️ Tecnologie Utilizzate](#-tecnologie-utilizzate)

- [🏗️ Architettura del Sistema Ideale](#-architettura-del-sistema-ideale)

- [🏗️ Architettura del Sistema Emulando gli SmartGlasses](#-architettura-del-sistema-emulando-gli-smartglasses)

- [⚡ Quickstart](#-quickstart)

- [🧠 Training del Modello](#-training-del-modello)

- [📡 API MQTT](#-api-mqtt)

- [🐳 Docker Deployment](#-docker-deployment)

- [📋 FURPS+](#-furps)

- [📐 Diagrammi UML](#-diagrammi-uml)

- [✅ CheckList Tecnologie Coinvolte](#-checklist-tecnologie-coinvolte)

- [⏱️ Diagramma di Gantt](#-diagramma-di-gantt)
  

---

  

## 🎯 Obiettivo

  

Il progetto **B.O.S.S.** mira a sviluppare un sistema di assistenza visiva basato su intelligenza artificiale per supportare persone con disabilità visive mediante degli occhiali da vista intelligenti. Utilizzando tecniche avanzate di Computer Vision, il sistema rileva e classifica oggetti domestici in tempo reale, fornendo feedback video per una navigazione sicura.

  

**Caratteristiche chiave:**

- 🎥 Rilevamento oggetti in tempo reale

- 📱 Interfaccia utente minimale per dispositivi wearable (AKA Smart Glasses)

- 🔄 Comunicazione asincrona via MQTT

- 🐳 Containerizzazione completa con Docker

- ⚡ Inferenza ottimizzata su GPU e CPU


---
  

## 🛠️ Tecnologie Utilizzate

### Artificial Intelligence and Learning Core

-  🧠 **PyTorch**: Framework per il deep learning, utilizzato per l'implementazione e l'addestramento del modello SSD300

-  👁 **TorchVision**: Libreria per computer vision, fornisce il modello SSD300 con backbone VGG16 pre-addestrato

-  📸 **OpenCV**: Per l'elaborazione di immagini e video, cattura da webcam e preprocessing

  

### Architettura Distribuita

-  📡 **MQTT (Mosquitto)**: Protocollo di messaggistica leggera per comunicazione basato su modello publisher/subscriber

-  🐳 **Docker & Docker Compose**: Containerizzazione per deployment scalabile e portabile

  

### Sviluppo e Deployment

-  🐍 **Python 3.13**: Linguaggio principale per tutti i componenti

-  🔢 **NumPy & Pillow**: Manipolazione di array e immagini

-  📨 **Paho-MQTT**: Libreria Python per MQTT

  

### Dataset

-  🏠 **COCO-Home-Objects**: Dataset espanso basato su COCO per oggetti domestici comuni

---


## 🏗️ Architettura del Sistema Ideale


```mermaid
flowchart TB

Client["SmartGlasses"] --> |"1. FRAME"| Broker["MQTT Broker"] --> |"2.FRAME"| Server["Server"] --> |"3.FRAME"| Inference["Model Inference"] --> |"4. BBOX"| Server --> |"5. BBOX"| Broker --> |"6. BBOX"| SmartGlasses["SmartGlasses GUI"]
SmartGlasses --> |"7. BBOX"| Client

```

---


## 🏗️ Architettura del Sistema Emulando gli SmartGlasses


```mermaid
flowchart TB

Client --> |"1. FRAME"| Broker["MQTT Broker"] --> |"2.FRAME"| Server["Server"] --> |"3.FRAME"| Inference["Model Inference"] --> |"4. BBOX"| Server --> |"5. BBOX"| Broker --> |"6. BBOX"| SmartGlasses
Client["Client"] --> |"1. FRAME"| SmartGlasses["SmartGlasses GUI"] --> |"2. FRAME"| Client
SmartGlasses --> |"7. FRAME + BBOX"| Client

```

---

### Componenti Dettagliati


#### 1. 🧠 **Modello SSD300-VGG16** 

-  **Single Shot MultiBox Detector (SSD)**: Architettura one-stage per object detection, fornisce in un solo passaggio di rete neurale sia le coordinate spaziali sia la classificazione degli oggetti riconosciuti

-  **VGG16 Backbone**: Rete convoluzionale pre-addestrata

-  **300x300 Input**: Risoluzione ottimale per velocità/inferenza bilanciata

-  **Output**: Bounding boxes, classi e punteggi di confidenza per ogni detection

  

#### 2. 🔍 **Servizio Inference** 

- Riceve immagini convertite in bytes via **MQTT**

- **Preprocessing**: resize a 300x300, normalizzazione

- **Inferenza**: passaggio attraverso il modello SSD

- **Post-processing**: thresholding delle predizioni

- **Output**: JSON con detections (bbox, class, confidence)

  

#### 3. 👓 **Client Wearable**

- Simulazione di **occhiali intelligenti**

- Cattura **frame** da video

- Invio al server via **MQTT**

- **Visualizzazione risultati**: overlay di bounding boxes



#### 4. 📡 **Broker MQTT** 

-  **Mosquitto**: Broker leggero e performante

- **Topics**: 
  - `smartglasses/frame` permette l'invio del frame, convertito in bytes, al server di inferenza
  - `smartglasses/pred` permette l'invio delle predizioni, formulate in una struttura JSON, al client wereable

- **QoS = 1**: Garantisce consegna almeno una volta nella comunicazione

  
---

  

## ⚡ Quickstart


Vuoi provare B.O.S.S. in 5 minuti? 🚀


```bash
# N.B. Per esecuzione su MacOS, richiede XQuartz, dopo l'installazione
# Per abilitare Connessioni client Apri XQuartz -> Impostazioni -> Protezione -> Consenti le Connessioni da Client Network
xhost +localhost

# 1. Clona la repository

git  clone  https://github.com/riccardosemeraro/B.O.S.S.-SSD300-VGG16.git

cd  B.O.S.S.-SSD300-VGG16

# 2. Build e Avvio tutto con Docker
docker-compose up --build
```

Il sistema sarà attivo utilizzando 3 container (`server`, `client`, `mqtt_broker`).

---
  

## 📦 Installazione Locale
  

### Prerequisiti

- Python 3.13

- Docker & Docker Compose

- GPU NVIDIA (opzionale)

- CONDA


```bash

# Crea ambiente virtuale
conda create -p ./.venv python=3.13 pip

# Attiva ambiente virtuale
conda activate ./.venv

# Installa dipendenze
pip install -r inference/requirements_inference.txt

# Avvio container mqtt
# in questo caso cambia BROKER_CONTAINER="localhost" invece di "mqtt_broker" in broker/configuration.py
docker run -d --name mosquitto -p 1883:1883 eclipse-mosquitto

# Avvio server
python3 server/server.py

# Avvio client
python3 client/client.py
```

  

### Configurazione Dataset

```bash

# Scarica il dataset COCO-Home-Objects al link 
# https://universe.roboflow.com/objectdetection-uzld5/coco-home-objects
# istruzioni in training/README_dataset.txt

```

---
  

## 🧠 Training del Modello
Il training del modello è stato eseguito mediante il
Jupyter Notebook in training/jupyter Google Colab

---

## 📡 API MQTT

  

### Topics

  
| Topic              | Direzione       | Payload              | Descrizione            |
|--------------------| --------------- | -------------------- | ---------------------- |
| smartglasses/frame | Client → Server | Base64 encoded image | Immagine da analizzare |
| smartglasses/pred  | Server → Client | JSON detections      | Risultati rilevamento  |


---
  

## 🐳 Docker Deployment

  

### Struttura Container

-  **server**: Servizio inference

-  **client**: GUI client con Tkinter

-  **mqtt_broker**: Mosquitto MQTT broker
  

### Build e Run

```bash

# Build e Avvio del docker-compose

docker-compose up -d --build

```

  

### Configurazione GPU

Per l'addestramento del modello si è ricorso a GOOGLE COLAB, il quale fornisce la GPU NVIDIA T4 per un periodo di 2/3 ore circa al giorno. Tramite addestramento basato su checkpoint si sono eseguite 100 epoche, di cui 20 sulla "testa" e 80 sul "corpo".

Per l'inferenze eseguita dal server, invece, si ricorre alla CPU APPLE SILICON M\*, ottimizzata per inferenze su singolo thread (non parallele). 

---

## 📋 FURPS+

  

### Funzionali (Functional)

-  **Rilevamento Oggetti**: Il sistema deve identificare oggetti domestici con accuratezza >50%

-  **Classificazione**: Assegnare classi corrette agli oggetti rilevati

-  **Bounding Boxes**: Fornire coordinate precise delle bounding boxes

-  **Real-time Processing**: Elaborare immagini

  

### Usabilità (Usability)

-  **Interfaccia Semplice**: GUI minimale accessibile per utenti con disabilità visive

-  **Feedback Visivo**: Overlay di bounding boxes chiare e leggibili

  

### Affidabilità (Reliability)

-  **Disponibilità**: Garantire la disponibilità del servizio

-  **Robustezza**: Gestione errori di rete e perdita connessione MQTT

-  **Accuratezza**: Mantenimento performance su dataset variati

  

### Performance (Performance)

-  **Velocità Inference**: <300ms per immagine 300x300

-  **Throughput**: Molto elevato

-  **Scalabilità**: Possibilità di scaling orizzontale con più server

  

### Supportabilità (Supportability)

-  **Manutenibilità**: Codice modulare e ben documentato

-  **Testabilità**: Suite di test completa per tutti i componenti

-  **Configurabilità**: Parametri regolabili via file di configurazione

-  **Monitoraggio**: Logging dettagliato per debugging

-  **Portabilità**: Supportato da diversi dispositivi grazie ai container

  

### + (Sicurezza, Privacy, etc.)

-  **Sicurezza**: Autenticazione MQTT opzionale

-  **Privacy**: Nessun storage permanente di immagini utente

-  **Compliance**: Adesione a GDPR per dati personali


---

## 📐 Diagrammi UML


### Diagramma di Sequenza

```mermaid

sequenceDiagram

participant Utente

participant Client

participant Broker MQTT

participant Server

participant Modello SSD

  

Utente->>Client: Cattura immagine

Client->>Broker MQTT: Pubblica immagine su 'smartglasses/frame'

Broker MQTT->>Server: Inoltra immagine

Server->>Modello SSD: Esegue inference

Modello SSD->>Server: Restituisce predizioni

Server->>Broker MQTT: Pubblica risultati su 'smartglasses/pred'

Broker MQTT->>Client: Inoltra predizioni

Client->>Utente: Visualizza bounding boxes

```

  

### Diagramma delle Classi

```mermaid
classDiagram

class Client {

-webcam

-display

+capture()

+send_image()

+show_results()

}

  

class MQTT{

-broker_address

-topics

+connect()

+publish()

+subscribe()

}

class Server {

-GPU

+send_inference()

}

  

class SSDModel {

-backbone

-weights

-layers

+train()

+predict()

}

  

Client --> MQTT : usa

Server --> MQTT : usa

Server --> SSDModel : carica
```

  

### Diagramma dei Casi d'Uso

```mermaid

flowchart LR
  User[Utente non vedente]
  Admin[Amministratore]
  System[Sistema B.O.S.S.]

  UC1([Cattura immagine])
  UC2([Invia al sistema B.O.S.S.])
  UC3([Visualizza Risultato])

  UC5([Carica modello])
  UC6([Elabora immagine ricevuta])
  UC7([Effettua inferenza])
  UC8([Invia risultati])

  User --> UC1 --> UC2 --> UC3
  System --> UC5 --> UC6 --> UC7 --> UC8
  Admin --> A1([Addestra modello])
  Admin --> A2([Monitora sistema])
  Admin --> A3([Aggiorna dataset])
  Admin --> A4([Raccoglie Feedback])
```


---

  

## ✅ CheckList Tecnologie Coinvolte

  

- [x] **Python 3.13**: Linguaggio principale

- [x] **PyTorch 2.8**: Framework deep learning

- [x] **TorchVision**: Modello SSD300 con Backbone VGG16 e utilities

- [x] **OpenCV**: Elaborazione immagini/video, GUI minimale

- [x] **MQTT (Mosquitto)**: Comunicazione distribuita, broker

- [x] **Paho-MQTT**: Libreria Python MQTT per Client e Server

- [x] **Docker & Docker Compose**: Containerizzazione

- [x] **COCO Dataset**: Dataset di training

- [x] **GPU NVIDIA**: Accelerazione CUDA (usata su Google Colab)

---

## ⏱️ Diagramma di Gantt

```mermaid
gantt
    title Gantt
    dateFormat  YYYY-MM-DD
    axisFormat %d/%m/%y
    todayMarker off
    
    section Introduction
        Ricerca : a1, 2025-10-10, 15d
        Addestramento AI: a2, after a1, 30d
    
    section Testing
        Object Detection: p4, after a2, 10d
        Sort: p5, after p4, 5d
        Tkinter GUI : p2, after p5, 7d
        OpenCV GUI : p3, after p2, 7d
        MQTT : p1, after p3, 5d
    
    section Beta Release
        Unify Sistem : b1, after p1, 15d
        Documentation: after b1, 12d
    
```

---

---

  

*Creato con ❤️ per rendere il mondo più accessibile.*