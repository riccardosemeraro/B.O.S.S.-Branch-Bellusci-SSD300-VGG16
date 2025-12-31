import time
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class HUDState:
    object_detection_on: bool = True
    inference_ms: float = 0.0
    fps: float = 5.0
    battery_pct: int = 87
    wifi_on: bool = True


class SmartGlassesGUI:
    def __init__(self):
        self.state = HUDState()

        # Dimensioni lente e Ratio
        self.video_h, self.video_w = 1080, 1920
        self.video_ratio = self.video_h / self.video_w

        # Inizializzo le variabili per il calcolo degli FPS
        self._t_last = time.time()
        self._fps_ema = 0.0

        # Colori BGR e font
        self.PILL_BG = (0, 0, 0)
        self.PILL_FG = (255, 255, 255)
        self.PILL_FONT = cv2.FONT_HERSHEY_DUPLEX
        self.PILL_FONT_SCALE = 0.6
        self.PILL_THICKNESS = 1
        self.PILL_PAD_X = 8
        self.PILL_PAD_Y = 6

    def setInferenceTime(self, ms):
        self.state.inference_ms = ms

    @staticmethod
    def _fit_size(max_w, max_h, ratio_h_over_w):
        # Parto dalla larghezza massima disponibile.
        w = max_w
        # Calcolo l'altezza mantenendo il rapporto altezza/larghezza desiderato.
        h = int(w * ratio_h_over_w)

        # Se l'altezza così calcolata supera l'altezza massima consentita
        if h > max_h:
            # limito l'altezza al massimo.
            h = max_h
            # E ricalcolo la larghezza in base al rapporto, così da mantenere le proporzioni.
            w = int(h / ratio_h_over_w)

        # Restituisco larghezza e altezza, assicurandomi che siano almeno 1 pixel.
        return max(1, w), max(1, h)

    def _draw_pill(self, img, x, y, text):
        # Calcola padding da dare ai Pill, estraendo larghezza e altezza della scritta
        (tw, th), _ = cv2.getTextSize(text, self.PILL_FONT, self.PILL_FONT_SCALE, self.PILL_THICKNESS)
        rx1, ry1 = x, y
        rx2, ry2 = x + tw + self.PILL_PAD_X * 2, y + th + self.PILL_PAD_Y * 2

        # Sfondo SOLIDO (no blur), disegna il rettangolo di sfondo
        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), self.PILL_BG, -1)

        # Disegna Testo ANTI-ALIASED
        cv2.putText(img, text, (x + self.PILL_PAD_X, y + self.PILL_PAD_Y + th - 2),
                    self.PILL_FONT, self.PILL_FONT_SCALE, self.PILL_FG, self.PILL_THICKNESS, cv2.LINE_AA)

    def _draw_left_hud(self, img):
        # Coordinate dove disegnare i Pill
        x, y = 12, 12

        # Testo dei pill
        od = f"[Object Detection]: {'ON' if self.state.object_detection_on and not self.state.inference_ms == 0.0 else 'OFF'}"
        inf = f"[Inference Time]: {f'{self.state.inference_ms:.2f}' if self.state.object_detection_on else '-'} ms"
        fps = f"[FPS]: {f'{self.state.fps:.0f}' if self.state.object_detection_on else '-'} fps"

        # Pill di loading se il modello non è ancora operativo
        if self.state.inference_ms == 0.0:
            text = "Loading model..."
            self._draw_pill(img, 380, 250, text)

        # Disegno i Pill sulla lente
        self._draw_pill(img, x, y, od)
        self._draw_pill(img, x, y + 35, inf)
        self._draw_pill(img, x, y + 70, fps)

    def _draw_right_hud(self, img, w):
        y = 12

        # Testo dei pill di destra
        bat = f"[BATTERY]: {self.state.battery_pct}%"
        wifi = f"[Wi-Fi]: {'ON' if self.state.wifi_on else 'OFF'}"
        bt = f"[Bluetooth]: {'ON' if self.state.object_detection_on else 'OFF'}"

        # Ricavo larghezza dei pill, considerando stile di sistema, e mostro su un'unica riga
        (tw1, th1), _ = cv2.getTextSize(bat, self.PILL_FONT, self.PILL_FONT_SCALE, self.PILL_THICKNESS)
        (tw2, th2), _ = cv2.getTextSize(wifi, self.PILL_FONT, self.PILL_FONT_SCALE, self.PILL_THICKNESS)
        (tw3, th3), _ = cv2.getTextSize(bt, self.PILL_FONT, self.PILL_FONT_SCALE, self.PILL_THICKNESS)

        # Calcolo inizio riga
        total_w = tw1 + tw2 + tw3 + self.PILL_PAD_X * 6 + 8
        start_x = max(12, w - 24 - total_w)

        # Pill di loading se il modello non è ancora operativo
        if self.state.inference_ms == 0.0:
            text = "Loading model..."
            self._draw_pill(img, start_x, 250, text)

        # Disegno i Pill sulla lente
        self._draw_pill(img, start_x, y, bat)
        self._draw_pill(img, start_x + tw1 + self.PILL_PAD_X * 2 + 4, y, wifi)
        self._draw_pill(img, start_x + tw1 + tw2 + self.PILL_PAD_X * 4 + 8, y, bt)

    def update_canvas(self, frame):

        # ---------------
        # CALCOLO FPS
        # ---------------
        t = time.time()  # Acquisisce il timestamp corrente in secondi

        # Calcola il delta tempo dal frame precedente, con minimo 1 microsecondo per evitare divisioni per zero.
        dt = max(1e-6, t - self._t_last)

        # Aggiorna il timestamp dell'ultimo frame per il prossimo ciclo.
        self._t_last = t

        # Calcola l'FPS istantaneo del frame corrente (1 / tempo impiegato)
        inst_fps = 1.0 / dt

        # Aggiorna la media mobile esponenziale (EMA): al primo frame usa l'istantaneo, altrimenti applica peso 90% alla media precedente + 10% all'istantaneo.
        self._fps_ema = inst_fps if self._fps_ema == 0 else (0.9 * self._fps_ema + 0.1 * inst_fps)

        # Salva la media EMA come FPS stabile nello stato dell'oggetto.
        self.state.fps = self._fps_ema

        # ----------------
        # GEOMETRIA LENTI
        # ----------------
        # Affiancamento lenti e dimensionamento
        h, w = frame.shape[:2]
        lens_w, lens_h = self._fit_size(w // 2 - 25, h, self.video_ratio)

        left_frame = cv2.resize(frame, (lens_w, lens_h))
        right_frame = cv2.resize(frame, (lens_w, lens_h))

        canvas_w = lens_w * 2 + 25
        canvas_h = lens_h
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        x1 = (lens_w - left_frame.shape[1]) // 2
        y1 = (lens_h - left_frame.shape[0]) // 2
        x2 = x1 + lens_w + 12
        canvas[y1:y1 + left_frame.shape[0], x1:x1 + left_frame.shape[1]] = left_frame
        canvas[y1:y1 + right_frame.shape[0], x2:x2 + right_frame.shape[1]] = right_frame

        # ----
        # PILL
        # ----
        # Inserimento dei PILL informativi
        self._draw_left_hud(canvas)
        self._draw_right_hud(canvas, canvas_w)

        return canvas