import numpy as np
import librosa

def compute_autocorrelation_f0(audio, sr):
    frame_length = int(sr * 0.025)
    hop_length = frame_length // 2

    if len(audio) < frame_length:
        return 0

    energy = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    zcr = librosa.feature.zero_crossing_rate(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    voiced_frames = np.where(energy > np.mean(energy))[0]

    f0_values = []

    for f in voiced_frames:
        start = f * hop_length
        end = start + frame_length
        frame = audio[start:end]

        if len(frame) < frame_length:
            continue

        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr) // 2:]

        d_min = sr // 500
        d_max = sr // 50

        if len(corr) > d_max and d_max > d_min:
            peak = np.argmax(corr[d_min:d_max]) + d_min
            if peak > 0:
                f0_values.append(sr / peak)

    return float(np.mean(f0_values)) if f0_values else 0
