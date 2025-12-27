import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass

# Se poi vuoi davvero renderizzare frame cv2 dentro Tkinter:
# pip install pillow
# from PIL import Image, ImageTk
# import cv2


@dataclass
class HUDState:
    object_detection_on: bool = True
    inference_ms: float = 200.0
    fps: float = 5.0
    battery_pct: int = 87
    wifi_on: bool = True


class SmartGlassesGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Glasses GUI")
        self.root.geometry("900x500")
        self.root.configure(bg="#000000")
        self.root.resizable(False, False)

        self.state = HUDState()

        # --- Area principale (due "cap" affiancate) ---
        self.main = tk.Frame(root, bg="#000000")
        self.main.pack(fill="both", expand=True, padx=12, pady=12)

        # Colonna sinistra: overlay list + cap sinistra
        self.left = tk.Frame(self.main, bg="#000000")
        self.left.pack(side="left", fill="both", expand=True)

        # Colonna destra: top badges + cap destra
        self.right = tk.Frame(self.main, bg="#000000")
        self.right.pack(side="right", fill="both", expand=True, padx=(12, 0))

        # --- Cap canvases (placeholders bianchi) ---
        self.cap_left = tk.Canvas(self.left, width=400, height=220, bg="#ffffff", highlightthickness=0)
        self.cap_left.pack(side="bottom", fill="x", expand=True)

        self.cap_right = tk.Canvas(self.right, width=400, height=220, bg="#ffffff", highlightthickness=0)
        self.cap_right.pack(side="bottom", fill="x", expand=True)


        # Colore pill
        self.PILL_BG = "#5AA7FF"

        # --- Overlay "pill" a sinistra (sopra al cap_left) ---
        # Tkinter non supporta vera trasparenza per widget; il "trasparente" è il bg del parent. [web:155]
        # Quindi: bg del contenitore = bg del canvas (bianco) e aggiungiamo gap con pady. [web:103]
        self.overlay_left = tk.Frame(self.left, bg=self.cap_left["bg"])
        self.overlay_left.place(x=12, y=12)

        gap = 8  # spazio tra pill (regola a gusto)

        self.lbl_od = self._pill(self.overlay_left, "👁️ ✅ Object Detection: ON")
        self.lbl_od.pack(anchor="w", pady=(0, gap))

        self.lbl_inf = self._pill(self.overlay_left, "⏱️ Inference Time: 200.00 ms")
        self.lbl_inf.pack(anchor="w", pady=(0, gap))

        self.lbl_fps = self._pill(self.overlay_left, "⚡ FPS: 5 fps")
        self.lbl_fps.pack(anchor="w", pady=0)

        # --- Badge in alto a destra (sopra al cap_right) ---
        # "Trasparenza" = bg del parent; usiamo il bianco del canvas e mettiamo gap con padx.
        self.overlay_right = tk.Frame(self.right, bg=self.cap_right["bg"])
        self.overlay_right.place(relx=1.0, x=-12, y=12, anchor="ne")

        gap_x = 8  # spazio tra i badge (regola a gusto)

        self.badge_batt = self._pill(self.overlay_right, f"🔋 Battery: {self.state.battery_pct}%", pad=(10, 4))
        self.badge_wifi = self._pill(self.overlay_right, "📶 Wi-Fi: ON", pad=(10, 4))
        self.badge_od   = self._pill(self.overlay_right, "🔵 Bluetooth: ON", pad=(10, 4))

        self.badge_batt.pack(side="left", padx=(0, gap_x), pady=0)
        self.badge_wifi.pack(side="left", padx=(0, gap_x), pady=0)
        self.badge_od.pack(side="left", padx=0, pady=0)

        self.apply_state()
        self.update_frames()

    def _pill(self, parent, text, pad=(12, 6)):
        # Stile simile allo screenshot: rettangolo blu, testo scuro.
        # Se vuoi la stessa identica palette, la aggiustiamo.
        bg = self.PILL_BG
        fg = "#001a33"
        return tk.Label(
            parent,
            text=text,
            bg=bg,
            fg=fg,
            font=("Helvetica", 10, "bold"),
            padx=pad[0],
            pady=pad[1],
        )

    def apply_state(self):
        self.lbl_od.config(
            text=f"👁️ {'✅' if self.state.object_detection_on else '❌'} Object Detection: "
                 f"{'ON' if self.state.object_detection_on else 'OFF'}"
        )
        self.lbl_inf.config(
            text=f"⏱️ Inference Time: "
                 f"{f'{self.state.inference_ms:.2f}' if self.state.object_detection_on else '-'} ms"
        )
        self.lbl_fps.config(
            text=f"⚡ FPS: {f'{self.state.fps:.0f}' if self.state.object_detection_on else '-'} fps"
        )

        self.badge_batt.config(text=f"🔋 Battery: {self.state.battery_pct}%")
        self.badge_wifi.config(text=f"📶 Wi-Fi: {'ON' if self.state.wifi_on else 'OFF'}")
        self.badge_od.config(text=f"🔵 Bluetooth: {'ON' if self.state.object_detection_on else 'OFF'}")

    def update_frames(self):
        """
        Qui ci attacchi cv2:
          ok, frame = cap.read()
          frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          img = ImageTk.PhotoImage(Image.fromarray(frame))
          canvas.create_image(0,0, anchor='nw', image=img)
          canvas.img = img  # keep reference
        """
        # Placeholder: aggiorna valori finti per vedere che l’HUD “vive”
        # (rimuovi quando colleghi i valori reali)
        # self.state.inference_ms = ...
        # self.state.fps = ...
        self.apply_state()
        self.root.after(100, self.update_frames)


if __name__ == "__main__":
    root = tk.Tk()
    app = SmartGlassesGUI(root)
    root.mainloop()
