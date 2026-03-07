import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

fs, signal = wav.read(r"C:\Users\LENOVO\OneDrive - ISTANBUL SAGLIK VE TEKNOLOJI UNIVERSITESI\Masaüstü\ses_kaydi.wav")
signal = signal / np.max(np.abs(signal)) 

win_len = int(0.02 * fs) 
hop_len = int(0.01 * fs)
frames_idx = range(0, len(signal) - win_len, hop_len)

energies = []
zcr_values = []

for i in frames_idx:
    frame = signal[i:i+win_len] * np.hamming(win_len) 
    energies.append(np.sum(frame**2)) # Enerji
    zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * win_len) # ZCR
    zcr_values.append(zcr)


noise_threshold = np.mean(energies[:int(0.2*fs/hop_len)]) * 1.5


output_signal = []
for i, e in enumerate(energies):
    if e > noise_threshold:
        
        start = frames_idx[i]
    
        output_signal.extend(signal[start : start + hop_len])


output_array = np.array(output_signal).astype(np.float32)
wav.write('shortened_output.wav', fs, output_array) 

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
ax1.plot(np.linspace(0, len(signal)/fs, len(signal)), signal)
ax1.set_title("Orijinal Ses Sinyali (Zaman Domaini)")

times_frames = np.array(frames_idx) / fs
ax2.plot(times_frames, energies, label="Enerji", color='blue')
ax2.axhline(y=noise_threshold, color='red', linestyle='--', label="Gürültü Eşiği")
ax2.set_title("Pencere Bazlı Enerji ve Eşik Değeri")
ax2.legend()

ax3.plot(np.linspace(0, len(signal)/fs, len(signal)), signal, color='gray', alpha=0.3)
for i, e in enumerate(energies):
    if e > noise_threshold:
        ax3.axvspan(frames_idx[i]/fs, (frames_idx[i]+win_len)/fs, color='green', alpha=0.3)
ax3.set_title("VAD (Konuşma Aktivitesi) Maskelenmiş Sinyal")

plt.tight_layout()
plt.savefig('odev_cikti_grafigi.png')

original_duration = len(signal) / fs
new_duration = len(output_array) / fs
print(f"Orijinal Süre: {original_duration:.2f} sn") 
print(f"Yeni Süre: {new_duration:.2f} sn") 
print(f"Sıkıştırma Oranı: %{((1 - new_duration/original_duration)*100):.2f}")