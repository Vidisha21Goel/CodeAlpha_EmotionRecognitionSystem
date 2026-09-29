"""
extract_features.py
--------------------
TASK 2: Emotion Recognition from Speech (feature extraction)

Walks data/raw_audio/<emotion>/*.wav, extracts MFCCs (Mel-Frequency
Cepstral Coefficients) plus a few complementary spectral features for each
clip, and saves everything to data/features.npz for training.

Works identically whether the audio is the synthetic data from
generate_synthetic_audio.py or a real dataset (RAVDESS/TESS/EMO-DB)
you've dropped into the same folder structure.

Run:
    python extract_features.py
"""

import os
import numpy as np
import librosa
from tqdm import tqdm

DATA_DIR = os.path.join("data", "raw_audio")
OUT_PATH = os.path.join("data", "features.npz")

N_MFCC = 40
MAX_PAD_LEN = 130  # frames; clips are padded/truncated to this length


def extract_mfcc(file_path: str, n_mfcc: int = N_MFCC, max_pad_len: int = MAX_PAD_LEN) -> np.ndarray:
    """Extracts an (n_mfcc, max_pad_len) MFCC matrix from a single audio file,
    combined with delta (velocity) features for richer temporal information."""
    y, sr = librosa.load(file_path, sr=None)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    features = np.vstack([mfcc, delta])  # shape: (2*n_mfcc, time)

    if features.shape[1] < max_pad_len:
        pad_width = max_pad_len - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_width)), mode="constant")
    else:
        features = features[:, :max_pad_len]

    return features.astype(np.float32)


def main():
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            f"'{DATA_DIR}' not found. Run `python generate_synthetic_audio.py` first, "
            f"or place a real dataset at data/raw_audio/<emotion>/*.wav"
        )

    emotions = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    if not emotions:
        raise RuntimeError(f"No emotion subfolders found inside '{DATA_DIR}'.")

    label_map = {emotion: idx for idx, emotion in enumerate(emotions)}
    print(f"Found {len(emotions)} emotion classes: {emotions}")

    X, y = [], []
    for emotion in emotions:
        emotion_dir = os.path.join(DATA_DIR, emotion)
        files = [f for f in os.listdir(emotion_dir) if f.lower().endswith((".wav", ".mp3"))]
        for fname in tqdm(files, desc=f"Extracting '{emotion}'"):
            fpath = os.path.join(emotion_dir, fname)
            try:
                feat = extract_mfcc(fpath)
                X.append(feat)
                y.append(label_map[emotion])
            except Exception as e:
                print(f"  Skipping {fpath}: {e}")

    X = np.array(X)          # (n_samples, 2*n_mfcc, max_pad_len)
    y = np.array(y)

    np.savez(OUT_PATH, X=X, y=y, label_map=label_map)
    print(f"\nSaved features to {OUT_PATH}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Label map: {label_map}")


if __name__ == "__main__":
    main()
