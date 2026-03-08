import numpy as np
import librosa

audio_path = "ses_kaydi.wav"
signal, fs = librosa.load(audio_path, sr=None)

signal = signal / np.max(np.abs(signal))

frame_duration = 0.02
hop_duration = 0.01

frame_length = int(frame_duration * fs)
hop_length = int(hop_duration * fs)

def framing(signal, frame_length, hop_length):
    frames = []
    for start in range(0, len(signal) - frame_length + 1, hop_length):
        frames.append(signal[start:start + frame_length])
    return np.array(frames)

def short_time_energy(frame):
    return np.sum(frame ** 2)

def zero_crossing_rate(frame):
    signs = np.sign(frame)
    signs[signs == 0] = 1
    return np.sum(np.abs(np.diff(signs))) / (2 * len(frame))

frames = framing(signal, frame_length, hop_length)

window = np.hamming(frame_length)
frames_windowed = frames * window

energy_values = np.array([short_time_energy(frame) for frame in frames_windowed])
zcr_values = np.array([zero_crossing_rate(frame) for frame in frames_windowed])

initial_silence_ms = 0.2
num_silence_frames = int((initial_silence_ms - frame_duration) / hop_duration) + 1
num_silence_frames = max(1, num_silence_frames)

noise_energy = np.mean(energy_values[:num_silence_frames])
vad_threshold = 2.0 * noise_energy

speech_mask = energy_values > vad_threshold

speech_energy = energy_values[speech_mask]
speech_zcr = zcr_values[speech_mask]

T_energy = np.median(speech_energy) if len(speech_energy) > 0 else 0
T_zcr = np.median(speech_zcr) if len(speech_zcr) > 0 else 0

labels = []
for i in range(len(frames_windowed)):
    if not speech_mask[i]:
        labels.append("silence")
    else:
        if energy_values[i] > T_energy and zcr_values[i] < T_zcr:
            labels.append("voiced")
        else:
            labels.append("unvoiced")