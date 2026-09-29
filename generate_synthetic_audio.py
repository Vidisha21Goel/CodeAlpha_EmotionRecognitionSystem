"""
generate_synthetic_audio.py
----------------------------
TASK 2: Emotion Recognition from Speech (data prep)

No dataset was provided with this assignment, and real speech-emotion
datasets (RAVDESS / TESS / EMO-DB) must be downloaded manually from
Kaggle/Zenodo since they're not available inside this environment.

This script generates SYNTHETIC .wav files that stand in for real speech,
with emotion-dependent pitch, tempo and energy characteristics baked in,
so the full pipeline (feature extraction -> CNN/LSTM training -> evaluation)
can be built, run and demonstrated end-to-end without a real dataset.

>>> FOR YOUR ACTUAL SUBMISSION <<<
Replace this synthetic data with a real dataset for a genuine result:
  1. RAVDESS (best documented): https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
  2. TESS:    https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
  3. EMO-DB:  http://emodb.bilderbar.info/download/

Just download one of these, extract it into data/raw_audio/<emotion>/*.wav
(one subfolder per emotion label) and skip this script - extract_features.py
will work on real audio the exact same way.

Run:
    python generate_synthetic_audio.py
"""

import os
import numpy as np
import soundfile as sf

RANDOM_STATE = 42
SR = 22050  # sample rate
DURATION = 2.5  # seconds
N_SAMPLES_PER_CLASS = 80

EMOTIONS = ["happy", "sad", "angry", "neutral", "fearful"]

# Rough emotion -> (base pitch Hz, vibrato depth, tempo/noise energy)
# These aren't scientifically accurate, just enough to give each class a
# distinguishable synthetic acoustic signature for the ML pipeline to learn.
EMOTION_PARAMS = {
    "happy":    {"pitch": 260, "vibrato": 18, "energy": 0.55, "tempo": 6.0},
    "sad":      {"pitch": 160, "vibrato": 4,  "energy": 0.25, "tempo": 2.0},
    "angry":    {"pitch": 220, "vibrato": 25, "energy": 0.85, "tempo": 7.5},
    "neutral":  {"pitch": 190, "vibrato": 6,  "energy": 0.35, "tempo": 3.5},
    "fearful":  {"pitch": 280, "vibrato": 30, "energy": 0.45, "tempo": 8.0},
}


def synthesize_utterance(emotion: str, rng: np.random.Generator) -> np.ndarray:
    params = EMOTION_PARAMS[emotion]
    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)

    base_pitch = params["pitch"] * (1 + rng.normal(0, 0.05))
    vibrato = params["vibrato"] * np.sin(2 * np.pi * params["tempo"] * t)
    freq = base_pitch + vibrato

    # A simple additive-synthesis "voice": fundamental + harmonics with
    # an amplitude envelope, so it has speech-like formant structure.
    phase = 2 * np.pi * np.cumsum(freq) / SR
    signal = (
        1.0 * np.sin(phase)
        + 0.5 * np.sin(2 * phase)
        + 0.25 * np.sin(3 * phase)
    )

    # Amplitude envelope: several "syllable" bursts
    n_syllables = rng.integers(3, 7)
    envelope = np.zeros_like(t)
    for _ in range(n_syllables):
        center = rng.uniform(0.1, DURATION - 0.1)
        width = rng.uniform(0.08, 0.2)
        envelope += np.exp(-((t - center) ** 2) / (2 * width ** 2))
    envelope = envelope / (envelope.max() + 1e-9)

    signal = signal * envelope * params["energy"]
    signal += rng.normal(0, 0.02, size=signal.shape)  # background noise
    signal = signal / (np.max(np.abs(signal)) + 1e-9) * 0.9
    return signal.astype(np.float32)


def main():
    rng = np.random.default_rng(RANDOM_STATE)
    out_root = os.path.join("data", "raw_audio")

    for emotion in EMOTIONS:
        emotion_dir = os.path.join(out_root, emotion)
        os.makedirs(emotion_dir, exist_ok=True)
        for i in range(N_SAMPLES_PER_CLASS):
            audio = synthesize_utterance(emotion, rng)
            path = os.path.join(emotion_dir, f"{emotion}_{i:03d}.wav")
            sf.write(path, audio, SR)
        print(f"Generated {N_SAMPLES_PER_CLASS} synthetic clips for '{emotion}'")

    total = len(EMOTIONS) * N_SAMPLES_PER_CLASS
    print(f"\nDone. {total} synthetic audio files saved under '{out_root}/'")


if __name__ == "__main__":
    main()
