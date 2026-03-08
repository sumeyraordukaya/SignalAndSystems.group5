import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog


root = tk.Tk()
root.withdraw()
print("Lütfen analiz edilecek 'ses_kaydi.wav' dosyasını seçin...")
input_path = filedialog.askopenfilename(title="Ses Dosyasını Seç", filetypes=[("WAV files", "*.wav")])

if not input_path:
    print("Dosya seçilmedi, program kapatılıyor.")
    exit()

output_path = os.path.join(os.path.dirname(input_path), 'shortened_output.wav')


fs, signal = wav.read(input_path)
signal = signal / np.max(np.abs(signal)) 


win_len = int(0.02 * fs) 
hop_len = int(0.01 * fs)
frames_idx = range(0, len(signal) - win_len, hop_len)

energies = []


for i in frames_idx:
    frame = signal[i:i+win_len] * np.hamming(win_len) 
    energies.append(np.sum(frame**2)) 


noise_threshold = np.mean(energies[:int(0.2*fs/hop_len)]) * 1.5


output_signal = []
for i, e in enumerate(energies):
    if e > noise_threshold:
        start = frames_idx[i]
        output_signal.extend(signal[start : start + hop_len])

output_array = np.array(output_signal).astype(np.float32)
wav.write(output_path, fs, output_array) 


fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12))


ax1.plot(np.linspace(0, len(signal)/fs, len(signal)), signal)
ax1.set_title("1. Orijinal Ses Sinyali (Zaman Domaini)")


times_frames = np.array(frames_idx) / fs
ax2.plot(times_frames, energies, label="Enerji", color='blue')
ax2.axhline(y=noise_threshold, color='red', linestyle='--', label="Gürültü Eşiği")
ax2.set_title("2. Enerji ve Dinamik Eşik Çizgisi")
ax2.legend()


ax3.plot(np.linspace(0, len(signal)/fs, len(signal)), signal, color='gray', alpha=0.3)
for i, e in enumerate(energies):
    if e > noise_threshold:
        ax3.axvspan(frames_idx[i]/fs, (frames_idx[i]+win_len)/fs, color='green', alpha=0.3)
ax3.set_title("3. VAD Maskesi (Yeşil: Konuşma)")


ax4.plot(np.linspace(0, len(output_array)/fs, len(output_array)), output_array, color='darkgreen')
ax4.set_title("4. ÇIKTI: Sessizliği Temizlenmiş Ses (Shortened Output)")

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(input_path), 'odev_cikti_grafigi.png'))


os.startfile(output_path) 

original_dur = len(signal) / fs
new_dur = len(output_array) / fs
print(f"Orijinal Süre: {original_dur:.2f} sn")
print(f"Yeni Süre: {new_dur:.2f} sn")
print(f"Sıkıştırma Oranı: %{((1 - new_dur/original_dur)*100):.2f}")


plt.show()