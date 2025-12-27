import tkinter as tk              # GUI toolkit
import os                         # path/file utilities
from dataclasses import dataclass # dataclass per stato HUD
import time                       # timing per FPS

from PIL import Image, ImageTk    # conversione frame -> immagine Tk
import cv2                        # OpenCV per leggere il video

# Directory dello script e path al video (relativo alla posizione del file .py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_VIDEO = os.path.join(BASE_DIR, "../inference/video4.mp4")

# --- "Manopole" di sizing finestra ---
VIDEO_SCALE_H = 0.35        # Altezza target della lente come frazione dell'altezza del video
MAX_SCREEN_H_FRAC = 0.80    # Non superare questa frazione dell'altezza dello schermo

# Stato dell'HUD (valori che cambieranno e che si riflettono sui pill)
@dataclass
class HUDState:
    object_detection_on: bool = True
    inference_ms: float = 200.0
    fps: float = 5.0
    battery_pct: int = 87
    wifi_on: bool = True


class SmartGlassesGUI:
    def __init__(self, root):
        # --- Root window ---
        self.root = root
        self.root.title("Smart Glasses GUI")
        self.root.configure(bg="#000000")  # sfondo generale
        self.root.resizable(True, True)    # finestra ridimensionabile

        # Stato HUD (modificabile runtime)
        self.state = HUDState()

        # --- VideoCapture ---
        self.cap = cv2.VideoCapture(PATH_VIDEO)
        if not self.cap.isOpened():
            raise RuntimeError(f"Impossibile aprire il video: {PATH_VIDEO}")

        # Dimensioni del video (usate per mantenere ratio)
        self.video_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        self.video_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)

        # Ratio espresso come (altezza / larghezza), utile per calcolare H da W (o viceversa)
        self.video_ratio = self.video_h / self.video_w if self.video_w else (9 / 16)

        # Imposta la geometry iniziale della finestra basandosi su video e schermo
        self._set_initial_geometry()

        # --- FPS (misurati sulla GUI) ---
        self._t_last = time.time()  # timestamp ultimo frame
        self._fps_ema = 0.0         # media mobile esponenziale FPS

        # --- Layout principale: due colonne (sinistra/destra) ---
        self.main = tk.Frame(root, bg="#000000")
        # fill="both" + expand=True => il frame cresce/si riduce con la finestra
        self.main.pack(fill="both", expand=True, padx=12, pady=12)

        # Colonna sinistra
        self.left = tk.Frame(self.main, bg="#000000")
        self.left.pack(side="left", fill="both", expand=True)

        # Colonna destra (con un gap a sinistra)
        self.right = tk.Frame(self.main, bg="#000000")
        self.right.pack(side="right", fill="both", expand=True, padx=(12, 0))

        # Viewport: area dove “centrerai” il canvas mantenendo il ratio
        self.left_view = tk.Frame(self.left, bg="#000000")
        self.left_view.pack(fill="both", expand=True)

        self.right_view = tk.Frame(self.right, bg="#000000")
        self.right_view.pack(fill="both", expand=True)

        # --- Canvas: rappresentano le due “lenti” ---
        # bg="#ffffff" è solo fallback: di fatto viene coperto dal frame video.
        self.cap_left = tk.Canvas(self.left_view, bg="#ffffff", highlightthickness=0)
        self.cap_right = tk.Canvas(self.right_view, bg="#ffffff", highlightthickness=0)

        # Usi place perché vuoi dimensionare e centrare manualmente i canvas al resize
        self.cap_left.place(x=0, y=0, width=10, height=10)
        self.cap_right.place(x=0, y=0, width=10, height=10)

        # --- Item immagine nel canvas (uno per canvas) ---
        # anchor="center" => l'immagine è centrata sul punto (x,y) che imposterai con coords(...)
        self._img_id_left = self.cap_left.create_image(0, 0, anchor="center")
        self._img_id_right = self.cap_right.create_image(0, 0, anchor="center")

        # Reference alle PhotoImage: serve tenerle vive (evita GC e “immagine che sparisce”)
        self._tk_img_left = None
        self._tk_img_right = None

        # --- HUD disegnato su Canvas (niente widget overlay => niente sfondi “sporgenti”) ---
        self.PILL_BG = "#5AA7FF"            # colore pill
        self.PILL_FG = "#001a33"            # colore testo
        self.PILL_FONT = ("Helvetica", 10, "bold")

        # HUD sinistra (tre pill in verticale)
        self.hud_left = self._create_left_hud(self.cap_left, x=12, y=12)

        # HUD destra (tre pill in orizzontale)
        self.hud_right = self._create_right_hud(self.cap_right, y=12)

        # Flag per evitare rientranze/loop durante resize
        self._resizing = False

        # Evento di resize: ricalcola dimensioni canvas e riposiziona HUD
        self.root.bind("<Configure>", self._on_resize)

        # Aggiorna i testi dei pill in base allo stato corrente
        self.apply_state()

        # Forza il layout (calcola correttamente winfo_width/height) e richiama un primo resize
        self.root.update_idletasks()
        self._on_resize(type("E", (), {"widget": self.root})())

        # Avvia il loop di update dei frame video
        self.update_frames()

    def _set_initial_geometry(self):
        """Calcola una geometry iniziale usando:
        - altezza desiderata della lente = video_h * VIDEO_SCALE_H
        - clamping sull'altezza schermo
        - larghezza coerente con due lenti affiancate + padding
        """
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()

        # Padding della UI (deriva dai pack padx/pady)
        pad_x = 12 * 2
        pad_y = 12 * 2
        col_gap = 12  # gap fra colonne (approx)

        # 1) Altezza lente basata sul video (es: metà dell'altezza originale)
        desired_lens_h = int(self.video_h * VIDEO_SCALE_H)

        # 2) Altezza finestra = lente + padding (questa NON usa lo schermo come base)
        target_h = desired_lens_h + pad_y

        # 3) Solo clamp di sicurezza sullo schermo
        target_h = min(target_h, int(screen_h * MAX_SCREEN_H_FRAC))

        # Altezza effettiva disponibile per la lente (tolti padding)
        lens_h = max(1, target_h - pad_y)

        # Con ratio (h/w) ricavi la larghezza necessaria per quella altezza
        lens_w = int(lens_h / self.video_ratio)

        # Due lenti affiancate + padding + gap colonne
        target_w = pad_x + (2 * lens_w) + col_gap

        # Clamping in larghezza (non superare quasi tutto lo schermo)
        target_w = min(target_w, int(screen_w * 0.95))

        # Applica geometry
        self.root.geometry(f"{target_w}x{target_h}")

    def _fit_size(self, max_w, max_h, ratio_h_over_w):
        """Calcola (w,h) che “fit” in (max_w,max_h) mantenendo un ratio (h/w)."""
        w = max_w
        h = int(w * ratio_h_over_w)
        if h > max_h:
            h = max_h
            w = int(h / ratio_h_over_w)
        return max(1, w), max(1, h)

    # --- Funzioni per creare/gestire i pill su canvas ---
    def _pill_items(self, canvas, x, y, text, pad_x=12, pad_y=6):
        """Crea un pill su canvas come:
        - text item (per misurare bbox)
        - rectangle dietro (con bordo nero)
        Ritorna dizionario con id del rettangolo e del testo.
        """
        # Crea testo (NW) con padding iniziale
        t = canvas.create_text(
            x + pad_x, y + pad_y,
            text=text,
            anchor="nw",
            fill=self.PILL_FG,
            font=self.PILL_FONT
        )

        # Bounding box del testo: (x1,y1,x2,y2)
        bbox = canvas.bbox(t)

        # Rettangolo dietro al testo con padding
        rx1 = bbox[0] - pad_x
        ry1 = bbox[1] - pad_y
        rx2 = bbox[2] + pad_x
        ry2 = bbox[3] + pad_y

        # outline="black" + width=1 => bordino nero
        r = canvas.create_rectangle(rx1, ry1, rx2, ry2, fill=self.PILL_BG, outline="black", width=1)

        # Metti il testo sopra al rettangolo
        canvas.tag_raise(t, r)

        return {"rect": r, "text": t}

    def _set_pill_text(self, canvas, pill, new_text, pad_x=12, pad_y=6):
        """Aggiorna il testo del pill e ridimensiona il rettangolo in base alla nuova bbox."""
        canvas.itemconfigure(pill["text"], text=new_text)
        bbox = canvas.bbox(pill["text"])
        canvas.coords(
            pill["rect"],
            bbox[0] - pad_x, bbox[1] - pad_y,
            bbox[2] + pad_x, bbox[3] + pad_y
        )

    def _move_pill(self, canvas, pill, x, y, pad_x=12, pad_y=6):
        """Sposta il pill (testo) alla posizione (x,y) e ridimensiona il rettangolo."""
        canvas.coords(pill["text"], x + pad_x, y + pad_y)
        bbox = canvas.bbox(pill["text"])
        canvas.coords(
            pill["rect"],
            bbox[0] - pad_x, bbox[1] - pad_y,
            bbox[2] + pad_x, bbox[3] + pad_y
        )

    def _create_left_hud(self, canvas, x, y):
        """Crea 3 pill verticali in alto a sinistra."""
        gap = 10
        p1 = self._pill_items(canvas, x, y, "👁️ ✅ Object Detection: ON")
        p2 = self._pill_items(canvas, x, y + (self._pill_height(canvas, p1) + gap), "⏱️ Inference Time: 200.00 ms")
        p3 = self._pill_items(canvas, x, y + (self._pill_height(canvas, p1) + gap) * 2, "⚡ FPS: 0 fps")
        return {"x": x, "y": y, "gap": gap, "pills": [p1, p2, p3]}

    def _create_right_hud(self, canvas, y):
        """Crea 3 pill orizzontali; x verrà calcolata al resize per allinearli a destra."""
        p1 = self._pill_items(canvas, 0, y, "🔋 Battery: 87%", pad_x=10, pad_y=4)
        p2 = self._pill_items(canvas, 0, y, "📶 Wi-Fi: ON", pad_x=10, pad_y=4)
        p3 = self._pill_items(canvas, 0, y, "🔵 Bluetooth: ON", pad_x=10, pad_y=4)
        gap_x = 10
        return {"y": y, "gap_x": gap_x, "pills": [p1, p2, p3]}

    def _pill_height(self, canvas, pill):
        """Altezza del rettangolo del pill."""
        x1, y1, x2, y2 = canvas.coords(pill["rect"])
        return int(y2 - y1)

    def _pill_width(self, canvas, pill):
        """Larghezza del rettangolo del pill."""
        x1, y1, x2, y2 = canvas.coords(pill["rect"])
        return int(x2 - x1)

    # --- Resize handler: centra canvas, mantiene ratio, riposiziona HUD ---
    def _on_resize(self, event):
        # Evita di processare resize per widget diversi dal root o durante un resize già in corso
        if event.widget is not self.root or self._resizing:
            return
        self._resizing = True

        # Assicura che le dimensioni calcolate da Tk siano aggiornate
        self.root.update_idletasks()

        # Dimensioni disponibili nei due viewport
        lw, lh = self.left_view.winfo_width(), self.left_view.winfo_height()
        rw, rh = self.right_view.winfo_width(), self.right_view.winfo_height()

        # Calcola dimensioni dei canvas rispettando il ratio del video
        w1, h1 = self._fit_size(max(1, lw), max(1, lh), self.video_ratio)
        w2, h2 = self._fit_size(max(1, rw), max(1, rh), self.video_ratio)

        # Centra ciascun canvas nel rispettivo viewport
        x1 = (lw - w1) // 2
        y1 = (lh - h1) // 2
        x2 = (rw - w2) // 2
        y2 = (rh - h2) // 2

        # Applica posizionamento e dimensioni canvas
        self.cap_left.place(x=x1, y=y1, width=w1, height=h1)
        self.cap_right.place(x=x2, y=y2, width=w2, height=h2)

        # Riposiziona i frame del video (item immagine) al centro dei canvas
        self.cap_left.coords(self._img_id_left, w1 // 2, h1 // 2)
        self.cap_right.coords(self._img_id_right, w2 // 2, h2 // 2)

        # HUD sinistra: stack verticale con gap fisso, ancorata top-left
        x = self.hud_left["x"]
        y = self.hud_left["y"]
        gap = self.hud_left["gap"]
        pills = self.hud_left["pills"]
        for i, p in enumerate(pills):
            self._move_pill(self.cap_left, p, x, y + i * (self._pill_height(self.cap_left, pills[0]) + gap))

        # HUD destra: allineamento top-right; calcola larghezza totale per partire a destra
        pills_r = self.hud_right["pills"]
        gap_x = self.hud_right["gap_x"]
        y_r = self.hud_right["y"]

        total_w = (
            self._pill_width(self.cap_right, pills_r[0]) + gap_x +
            self._pill_width(self.cap_right, pills_r[1]) + gap_x +
            self._pill_width(self.cap_right, pills_r[2])
        )

        # start_x scelto per finire a w2-12 (margine destro) senza andare negativo
        start_x = max(12, w2 - 12 - total_w)

        # Posiziona ogni pill in fila
        x_cur = start_x
        for p in pills_r:
            self._move_pill(self.cap_right, p, x_cur, y_r, pad_x=10, pad_y=4)
            x_cur += self._pill_width(self.cap_right, p) + gap_x

        self._resizing = False

    # --- Aggiornamento testi HUD ---
    def apply_state(self):
        # Costruisce stringhe in base allo stato
        od = f"👁️ {'✅' if self.state.object_detection_on else '❌'} Object Detection: {'ON' if self.state.object_detection_on else 'OFF'}"
        inf = f"⏱️ Inference Time: {f'{self.state.inference_ms:.2f}' if self.state.object_detection_on else '-'} ms"
        fps = f"⚡ FPS: {f'{self.state.fps:.0f}' if self.state.object_detection_on else '-'} fps"

        # Aggiorna i tre pill di sinistra
        self._set_pill_text(self.cap_left, self.hud_left["pills"][0], od)
        self._set_pill_text(self.cap_left, self.hud_left["pills"][1], inf)
        self._set_pill_text(self.cap_left, self.hud_left["pills"][2], fps)

        # Aggiorna i tre pill di destra
        self._set_pill_text(self.cap_right, self.hud_right["pills"][0], f"🔋 Battery: {self.state.battery_pct}% ", pad_x=10, pad_y=4)
        self._set_pill_text(self.cap_right, self.hud_right["pills"][1], f"📶 Wi-Fi: {'ON' if self.state.wifi_on else 'OFF'}", pad_x=10, pad_y=4)
        self._set_pill_text(self.cap_right, self.hud_right["pills"][2], f"🔵 Bluetooth: {'ON' if self.state.object_detection_on else 'OFF'}", pad_x=10, pad_y=4)

    # --- Rendering frame -> canvas ---
    def _render_on_canvas(self, canvas, img_id, frame_bgr, target_w, target_h):
        # Converti BGR (OpenCV) -> RGB (PIL/Tk)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Converti in PIL image e ridimensiona alle dimensioni del canvas
        pil_img = Image.fromarray(frame_rgb).resize((target_w, target_h), Image.LANCZOS)

        # Converte in PhotoImage per Tk
        tk_img = ImageTk.PhotoImage(pil_img)

        # Aggancia la PhotoImage all'item del canvas
        canvas.itemconfigure(img_id, image=tk_img)

        # Ritorna reference per evitare garbage collection
        return tk_img

    # --- Loop di update frame (scheduling con after) ---
    def update_frames(self):
        # Legge un frame dal video
        ok, frame = self.cap.read()

        # Se finito, rewind e riprova (loop)
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok:
                # Se ancora fallisce, ritenta dopo 200ms
                self.root.after(200, self.update_frames)
                return

        # Calcolo FPS GUI: EMA su FPS istantaneo
        t = time.time()
        dt = max(1e-6, t - self._t_last)
        self._t_last = t
        inst_fps = 1.0 / dt
        self._fps_ema = inst_fps if self._fps_ema == 0 else (0.9 * self._fps_ema + 0.1 * inst_fps)
        self.state.fps = self._fps_ema

        # Dimensioni attuali dei canvas
        w1, h1 = self.cap_left.winfo_width(), self.cap_left.winfo_height()
        w2, h2 = self.cap_right.winfo_width(), self.cap_right.winfo_height()

        # Se non pronti (fase iniziale), riprova
        if w1 <= 1 or h1 <= 1 or w2 <= 1 or h2 <= 1:
            self.apply_state()
            self.root.after(30, self.update_frames)
            return
        
        # draw boxes

        # Renderizza lo stesso frame su entrambe le lenti
        self._tk_img_left = self._render_on_canvas(self.cap_left, self._img_id_left, frame, w1, h1)
        self._tk_img_right = self._render_on_canvas(self.cap_right, self._img_id_right, frame, w2, h2)

        # Aggiorna testi HUD (FPS ecc.)
        self.apply_state()

        # Pianifica prossimo frame (circa ~66 fps teorici, dipende dal carico)
        self.root.after(15, self.update_frames)


if __name__ == "__main__":
    # Entry point: crea root, istanzia GUI, avvia loop eventi Tk
    root = tk.Tk()
    app = SmartGlassesGUI(root)
    root.mainloop()