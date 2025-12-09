FROM python:3.13-slim

# Installa build dependencies per sort.py
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    libx11-6 \
    libxkbcommon-x11-0 \
    libxcb1 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-sync1 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libgtk-3-0 \
    libqt5gui5 \
    libqt5core5a \
    libqt5widgets5

WORKDIR /app

# Copia requirements dentro il container
COPY requirements_inference.txt .
RUN pip install -r requirements_inference.txt

# Comando di avvio
CMD ["python", "inference/inference_video.py"]
