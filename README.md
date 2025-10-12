# progetto B.O.S.S. (Blind Oriented Suggested System)
Groppo Progetto SAPD: Miki Palmisano, Riccardo Semeraro

Gruppo: Tiziano Albore, Alessio Lorè, Gabriele Martemucci

Obiettivo: progettare e ottimizzare reti di Computer Vision dedicate all’individuazione e classificazione di ostacoli all’interno di immagini o video, al fine di supportare la navigazione autonoma e la sicurezza del sistema.

From Branch "Bellusci-SSD300-VGG16" of the original repo: "https://github.com/sickcrash/B.O.S.S.-Albore-Lore-Martemucci"

---

# 🎯 Obiettivo della tesi:
- Clonare e riorganizzare la repository esistente di training del modello SSD300.
- Realizzare uno script di inference, che a partire da un modello addestrato produca bounding box, classi e confidenze sugli oggetti rilevati.
- Integrare l’inference in un servizio backend che comunichi tramite MQTT.
- Simulare il comportamento di un wearable (occhiali smart) tramite un client Docker con GUI minimale, che invii immagini al server e visualizzi le predizioni ricevute (immagine + box + classi + confidenze).

---

# 📂 Struttura della repository (proposta)
Dopo la riorganizzazione, la repository dovrà avere questa struttura modulare:

project/<br>
│<br>
├── training/<br>
├── inference/<br>
├── server/               # servizio che si sottoscrive al broker MQTT ed esegue inference <br>
├── client/               # simulazione wearable: invio immagine + GUI minimale <br>
├── broker/               # configurazione del broker MQTT (es. Mosquitto) <br>
├── saved_models/         # modelli addestrati salvati <br>
├── docker/               # docker-compose con client, server e broker│ <br>
├── requirements.txt <br>
└── README.md <br>
