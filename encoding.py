import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import os

fs = 44100
sure = 0.04  

frekanslar = {
    'A': (400, 1000), 'B': (400, 1100), 'C': (400, 1200), 'Ç': (400, 1300),
    'D': (500, 1000), 'E': (500, 1100), 'F': (500, 1200), 'G': (500, 1300),
    'Ğ': (600, 1000), 'H': (600, 1100), 'I': (600, 1200), 'İ': (600, 1300),
    'J': (700, 1000), 'K': (700, 1100), 'L': (700, 1200), 'M': (700, 1300),
    'N': (800, 1000), 'O': (800, 1100), 'Ö': (800, 1200), 'P': (800, 1300),
    'R': (900, 1000), 'S': (900, 1100), 'Ş': (900, 1200), 'T': (900, 1300),
    'U': (300, 1000), 'Ü': (300, 1100), 'V': (300, 1200), 'Y': (300, 1300),
    'Z': (200, 1000), ' ': (200, 1100)  # Boşluk karakteri
}

def harf_sinyali_uret(f1, f2):
    t = np.linspace(0, sure, int(fs * sure), endpoint=False)
    # s(t) = sin(2pi*f1*t) + sin(2pi*f2*t)
    sinyal = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)
    return sinyal

def metni_sese_donustur(metin):
    metin_sinyali = []
    
    for harf in metin.upper():
        if harf in frekanslar:
            f1, f2 = frekanslar[harf]
            sinyal = harf_sinyali_uret(f1, f2)
            metin_sinyali.extend(sinyal)
       
            sessizlik = np.zeros(int(fs * 0.005))
            metin_sinyali.extend(sessizlik)
        else:
            print(f"Uyarı: '{harf}' karakteri tabloda yok!")

  
    son_sessizlik = np.zeros(int(fs * 0.050))
    metin_sinyali.extend(son_sessizlik)

    return np.array(metin_sinyali) 

mesaj = "afiyet olsun"
cikti_sinyali = metni_sese_donustur(mesaj)

print(f"'{mesaj}' çalınıyor...")
sd.play(cikti_sinyali, fs)
sd.wait()

max_val = np.max(np.abs(cikti_sinyali))
if max_val > 0:
    normalize_sinyal = cikti_sinyali / max_val
    yazilacak_veri = np.int16(normalize_sinyal * 32767)
    
    dosya_adi = "uzun_mesaj.wav"
    wav.write(dosya_adi, fs, yazilacak_veri)
    print(f"'{dosya_adi}' kaydedildi.")
    os.startfile(dosya_adi)