import numpy as np
import librosa


def extract_features(audio, sr):
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)

    if len(audio) < frame_length:
        return None, None, None

    frames = librosa.util.frame(
        audio,
        frame_length=frame_length,
        hop_length=hop_length
    )

    energy = np.sum(frames ** 2, axis=0)

    zcr = librosa.feature.zero_crossing_rate(
        audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    energy_threshold = np.mean(energy) * 0.5
    voiced_frames = energy > energy_threshold

    return energy, zcr, voiced_frames