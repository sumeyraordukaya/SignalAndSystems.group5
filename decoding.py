import numpy as np
import scipy.io.wavfile as wav


fs = 44100
sure = 0.04  
pencere_boyutu = int(fs * sure)


frekanslar = {
    'A': (400, 1000), 'B': (400, 1100), 'C': (400, 1200), 'Ç': (400, 1300),
    'D': (500, 1000), 'E': (500, 1100), 'F': (500, 1200), 'G': (500, 1300),
    'Ğ': (600, 1000), 'H': (600, 1100), 'I': (600, 1200), 'İ': (600, 1300),
    'J': (700, 1000), 'K': (700, 1100), 'L': (700, 1200), 'M': (700, 1300),
    'N': (800, 1000), 'O': (800, 1100), 'Ö': (800, 1200), 'P': (800, 1300),
    'R': (900, 1000), 'S': (900, 1100), 'Ş': (900, 1200), 'T': (900, 1300),
    'U': (300, 1000), 'Ü': (300, 1100), 'V': (300, 1200), 'Y': (300, 1300),
    'Z': (200, 1000), ' ': (200, 1100)
}

def goertzel(sinyal, hedef_frekans, fs):
    
    N = len(sinyal)
    k = int(0.5 + (N * hedef_frekans) / fs)
    w = (2 * np.pi / N) * k
    cosine = np.cos(w)
    coeff = 2 * cosine
    
    s_prev = 0
    s_prev2 = 0
    for x in sinyal:
        s = x + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s
    
    guc = s_prev2**2 + s_prev**2 - coeff * s_prev * s_prev2
    return guc

def sesi_coz(dosya_yolu):
    ornek_hizi, veri = wav.read(dosya_yolu)
   
    if len(veri.shape) > 1: veri = veri[:, 0]
    
   
    veri = veri / np.max(np.abs(veri))
    
    cozulen_metin = ""
    son_harf = ""

  
    for i in range(0, len(veri), pencere_boyutu):
        pencere = veri[i : i + pencere_boyutu]
        if len(pencere) < pencere_boyutu: break
        
        
        pencere = pencere * np.hamming(len(pencere))
        
        en_iyi_harf = None
        en_yuksek_guc = 0
        
     
        for harf, (f1, f2) in frekanslar.items():
            guc = goertzel(pencere, f1, fs) + goertzel(pencere, f2, fs)
            if guc > en_yuksek_guc:
                en_yuksek_guc = guc
                en_iyi_harf = harf
        
        if en_yuksek_guc > 10:
            if en_iyi_harf != son_harf:
                cozulen_metin += en_iyi_harf
                son_harf = en_iyi_harf
        else:
            
            son_harf = ""

    return cozulen_metin

# Kullanım:
print("Çözülen Mesaj:", sesi_coz("uzun_mesaj.wav"))