import tkinter as tk
from tkinter import ttk, Canvas
import math
import random
import time


class SmartGlassesLens:
    def __init__(self, root):
        self.root = root
        self.root.title("🕶️ Smart Glasses Lens Simulator")
        self.root.geometry("900x700")
        self.root.configure(bg='#0a0a0a')
        self.root.resizable(False, False)

        # Canvas principale per la lente
        self.canvas = Canvas(root, width=800, height=500, bg='#1a1a2e',
                             highlightthickness=0, relief='flat')
        self.canvas.pack(pady=30)

        # HUD Info Panel
        self.hud_frame = tk.Frame(root, bg='#0a0a0a', height=150)
        self.hud_frame.pack(fill='x', padx=20)
        self.hud_frame.pack_propagate(False)

        # Indicatori HUD
        self.status_label = tk.Label(self.hud_frame, text="🔋 87% | 👁️ Object Detection: ON | 📶 WiFi",
                                     fg='#00ff88', bg='#0a0a0a', font=('Courier', 12))
        self.status_label.pack(pady=10)

        self.detection_label = tk.Label(self.hud_frame, text="👤 Persona rilevata (87% conf.) | 🚗 Auto (92%)",
                                        fg='#ffaa00', bg='#0a0a0a', font=('Courier', 11))
        self.detection_label.pack()

        # Controlli
        self.controls = ttk.Frame(root)
        self.controls.pack(pady=10)

        ttk.Button(self.controls, text="▶️ Avvia AR", command=self.start_ar).pack(side='left', padx=5)
        ttk.Button(self.controls, text="⏸️ Pausa", command=self.pause_ar).pack(side='left', padx=5)
        ttk.Button(self.controls, text="🎨 Modalità Notte", command=self.night_mode).pack(side='left', padx=5)

        # Variabili animazione
        self.running = False
        self.night_mode = False
        self.particles = []
        self.objects = []
        self.time = 0

        self.create_lens()
        self.animate()

    def create_lens(self):
        """Crea la lente con effetti vetro e frame"""
        canvas = self.canvas

        # Frame occhiale destro (metallico)
        canvas.create_oval(50, 50, 350, 350, fill='#333', outline='#666', width=4)
        canvas.create_oval(60, 60, 340, 340, fill='#444', outline='#888', width=2)

        # Riflessi sul frame
        canvas.create_arc(70, 70, 330, 330, start=45, extent=60, fill='#777', outline='')
        canvas.create_arc(70, 70, 330, 330, start=200, extent=60, fill='#777', outline='')

        # Lente vetro con gradiente e riflessi
        # Vetro principale (trasparente simulato)
        canvas.create_oval(80, 80, 320, 320, fill='#88c0d0', outline='', stipple='gray25')

        # Riflesso principale lente
        canvas.create_oval(100, 100, 280, 280, fill='#aaddff', outline='', stipple='gray12')
        canvas.create_oval(120, 120, 260, 260, fill='#bbddff', outline='', stipple='gray25')

        # Effetto curvatura lente
        for i in range(0, 360, 30):
            angle = math.radians(i)
            x1 = 200 + 120 * math.cos(angle)
            y1 = 200 + 120 * math.sin(angle)
            x2 = 200 + 130 * math.cos(angle)
            y2 = 200 + 130 * math.sin(angle)
            canvas.create_line(x1, y1, x2, y2, fill='#66aacc', width=1)

    def draw_hud_overlay(self):
        """Overlay HUD con elementi AR"""
        canvas = self.canvas
        self.time += 0.1

        # Grid HUD
        for i in range(0, 360, 30):
            angle = math.radians(i + self.time)
            x1 = 200 + 140 * math.cos(angle)
            y1 = 200 + 140 * math.sin(angle)
            x2 = 200 + 150 * math.cos(angle)
            y2 = 200 + 150 * math.sin(angle)
            color = '#00ff88' if self.night_mode else '#00aa66'
            canvas.create_line(x1, y1, x2, y2, fill=color, width=1)

        # Indicatori AR (bounding box simulate)
        if random.random() < 0.7:
            # Persona
            x, y, w, h = 180 + 20 * math.sin(self.time), 160 + 20 * math.cos(self.time), 80, 120
            canvas.create_rectangle(x, y, x + w, y + h, outline='#ff4444', width=3)
            canvas.create_text(x + w / 2, y - 15, text="👤 PERSONA", fill='#ff4444', font=('Courier', 10, 'bold'))

        if random.random() < 0.5:
            # Auto
            x2, y2, w2, h2 = 250 + 15 * math.sin(self.time * 1.3), 220, 60, 40
            canvas.create_rectangle(x2, y2, x2 + w2, y2 + h2, outline='#44ff44', width=2)
            canvas.create_text(x2 + w2 / 2, y2 - 10, text="🚗 AUTO", fill='#44ff44', font=('Courier', 9, 'bold'))

        # Particelle floating (polvere/effetti)
        self.particles = [(random.randint(100, 300), random.randint(100, 300)) for _ in range(15)]
        for px, py in self.particles:
            alpha = 0.3 + 0.3 * math.sin(self.time + px * 0.01)
            size = 2 + alpha * 3
            canvas.create_oval(px, py, px + size, py + size, fill='#88ccff', outline='')

    def start_ar(self):
        self.running = True
        self.detection_label.config(text="🔍 AR attiva - Rilevamento in tempo reale")

    def pause_ar(self):
        self.running = False
        self.detection_label.config(text="⏸️ AR in pausa")

    def night_mode(self):
        self.night_mode = not self.night_mode
        bg_color = '#000011' if self.night_mode else '#1a1a2e'
        self.canvas.configure(bg=bg_color)
        self.status_label.config(
            text=f"🔋 87% | 👁️ Object Detection: ON | 📶 WiFi | 🌙 {'ON' if self.night_mode else 'OFF'}")

    def animate(self):
        canvas = self.canvas
        canvas.delete("hud")  # Pulisci solo HUD

        if self.running:
            self.draw_hud_overlay()

        self.root.after(50, self.animate)  # 20 FPS


if __name__ == "__main__":
    root = tk.Tk()
    app = SmartGlassesLens(root)
    root.mainloop()
