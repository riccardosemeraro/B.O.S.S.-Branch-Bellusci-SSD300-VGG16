import os
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

    # "Manopole" di sizing finestra
    VIDEO_SCALE_H = 0.35
    MAX_SCREEN_H_FRAC = 0.80


class SmartGlassesGUI:
    def __init__(self):
        self.state = HUDState()

        self.video_h, self.video_w = 1080, 1920 #self.state.frame.shape[:2]
        self.video_ratio = self.video_h / self.video_w

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
        self.PILL_RADIUS = 14

        self._set_initial_geometry()

    def setInferenceTime(self, ms):
        self.state.inference_ms = ms

    def _set_initial_geometry(self):
        screen_w = 1920
        screen_h = 1080
        pad_x = 24
        pad_y = 24
        col_gap = 12

        desired_lens_h = int(self.video_h * self.state.VIDEO_SCALE_H)
        target_h = desired_lens_h + pad_y
        target_h = min(target_h, int(screen_h * self.state.MAX_SCREEN_H_FRAC))
        lens_h = max(1, target_h - pad_y)
        lens_w = int(lens_h / self.video_ratio)
        target_w = pad_x + (2 * lens_w) + col_gap
        target_w = min(target_w, int(screen_w * 0.95))

        cv2.resizeWindow("Smart Glasses GUI", target_w, target_h)

    @staticmethod
    def _fit_size(max_w, max_h, ratio_h_over_w):
        w = max_w
        h = int(w * ratio_h_over_w)
        if h > max_h:
            h = max_h
            w = int(h / ratio_h_over_w)
        return max(1, w), max(1, h)

    def _draw_pill(self, img, x, y, text):
        (tw, th), baseline = cv2.getTextSize(text, self.PILL_FONT, self.PILL_FONT_SCALE, self.PILL_THICKNESS)
        rx1, ry1 = x, y
        rx2, ry2 = x + tw + self.PILL_PAD_X * 2, y + th + self.PILL_PAD_Y * 2

        # Sfondo SOLIDO (no blur)
        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), self.PILL_BG, -1)

        # Bordo netto
        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (60, 60, 60), 1)

        # Testo ANTI-ALIASED
        cv2.putText(img, text, (x + self.PILL_PAD_X, y + self.PILL_PAD_Y + th - 2),
                    self.PILL_FONT, self.PILL_FONT_SCALE, self.PILL_FG, self.PILL_THICKNESS, cv2.LINE_AA)

    def _draw_left_hud(self, img):
        x, y = 12, 12

        od = f"[Object Detection]: {'ON' if self.state.object_detection_on else 'OFF'}"
        inf = f"[Inference Time]: {f'{self.state.inference_ms:.2f}' if self.state.object_detection_on else '-'} ms"
        fps = f"[FPS]: {f'{self.state.fps:.0f}' if self.state.object_detection_on else '-'} fps"

        self._draw_pill(img, x, y, od)
        self._draw_pill(img, x, y + 35, inf)
        self._draw_pill(img, x, y + 70, fps)

    def _draw_right_hud(self, img, w):
        y = 12
        bat = f"[BATTERY]: {self.state.battery_pct}%"
        wifi = f"[Wi-Fi]: {'ON' if self.state.wifi_on else 'OFF'}"
        bt = f"[Bluetooth]: {'ON' if self.state.object_detection_on else 'OFF'}"

        (tw1, th1), _ = cv2.getTextSize(bat, self.PILL_FONT, self.PILL_FONT_SCALE, self.PILL_THICKNESS)
        (tw2, th2), _ = cv2.getTextSize(wifi, self.PILL_FONT, self.PILL_FONT_SCALE, self.PILL_THICKNESS)
        (tw3, th3), _ = cv2.getTextSize(bt, self.PILL_FONT, self.PILL_FONT_SCALE, self.PILL_THICKNESS)

        total_w = tw1 + tw2 + tw3 + self.PILL_PAD_X * 6 + 8
        start_x = max(12, w - 24 - total_w)

        self._draw_pill(img, start_x, y, bat)
        self._draw_pill(img, start_x + tw1 + self.PILL_PAD_X * 2 + 4, y, wifi)
        self._draw_pill(img, start_x + tw1 + tw2 + self.PILL_PAD_X * 4 + 8, y, bt)

    def update_canvas(self, frame):

        t = time.time()
        dt = max(1e-6, t - self._t_last)
        self._t_last = t
        inst_fps = 1.0 / dt
        self._fps_ema = inst_fps if self._fps_ema == 0 else (0.9 * self._fps_ema + 0.1 * inst_fps)
        self.state.fps = self._fps_ema

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

        self._draw_left_hud(canvas)
        self._draw_right_hud(canvas, canvas_w)

        return canvas