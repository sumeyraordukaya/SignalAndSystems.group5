import tkinter as tk
from tkinter import messagebox
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class DTMFApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DTMF Signal Synthesizer - Group 5 ")
        self.dtmf_map = {
            '1': (697, 1209), '2': (697, 1336), '3': (697, 1477), 'A': (697, 1633),
            '4': (770, 1209), '5': (770, 1336), '6': (770, 1477), 'B': (770, 1633),
            '7': (852, 1209), '8': (852, 1336), '9': (852, 1477), 'C': (852, 1633),
            '*': (941, 1209), '0': (941, 1336), '#': (941, 1477), 'D': (941, 1633)
        }
        
        self.fs = 44100  
        self.duration = 0.3  
        self.setup_ui()
    def setup_ui(self):
        keypad_frame = tk.Frame(self.root)
        keypad_frame.pack(side=tk.LEFT, padx=20, pady=20)

        keys = [
            ['1', '2', '3', 'A'],
            ['4', '5', '6', 'B'],
            ['7', '8', '9', 'C'],
            ['*', '0', '#', 'D']
        ]

        for r, row in enumerate(keys):
            for c, key in enumerate(row):
                btn = tk.Button(keypad_frame, text=key, width=10, height=3,
                                command=lambda k=key: self.on_key_press(k))
                btn.grid(row=r, column=c, padx=5, pady=5)

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.ax.set_title("Sinyal Grafiği (Zaman Etki Alanı)")
        self.ax.set_xlabel("Zaman (s)")
        self.ax.set_ylabel("Genlik")

    def on_key_press(self, key):
        f_low, f_high = self.dtmf_map[key]
        
        t = np.linspace(0, self.duration, int(self.fs * self.duration), endpoint=False)
        
        signal = 0.5 * (np.sin(2 * np.pi * f_low * t) + np.sin(2 * np.pi * f_high * t))
        
        sd.play(signal, self.fs)
        
        self.ax.clear()
        mask = t <= 0.02  
        self.ax.plot(t[mask], signal[mask], color='orange')
        self.ax.set_title(f"'{key}' Tuşu Sinyali ({f_low}Hz + {f_high}Hz)")
        self.ax.set_xlabel("Zaman (s)")
        self.ax.set_ylabel("Genlik")
        self.ax.grid(True)
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = DTMFApp(root)
    root.mainloop()