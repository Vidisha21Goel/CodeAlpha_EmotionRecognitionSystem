# CodeAlpha_EmotionRecognitionSystem

**Task:** Recognize human emotions (happy, angry, sad, neutral, fearful) from speech audio.

## Overview

This project builds a **hybrid CNN + LSTM deep learning pipeline** for
speech emotion recognition:
1. **Feature extraction:** MFCCs (Mel-Frequency Cepstral Coefficients) +
   delta features from each audio clip.
2. **Model:** Conv2D layers learn local spectral patterns across the MFCC
   coefficients, an LSTM layer learns temporal dynamics across frames,
   followed by dense layers for classification.
3. **Evaluation:** Accuracy, Precision, Recall, F1-score (macro-averaged),
   confusion matrix, training curves.

## About the dataset — IMPORTANT

**No dataset was provided with this task**, and real datasets like
RAVDESS/TESS/EMO-DB are not downloadable from inside a sandboxed dev
environment, so `generate_synthetic_audio.py` **synthesizes placeholder
.wav files** with emotion-dependent pitch/tempo/energy signatures. This
lets the entire pipeline (extraction → training → evaluation) run and be
demonstrated end-to-end with zero setup.

**⚠️ For your actual submission, replace this with a real dataset** — the
synthetic audio is acoustically simple, so the model hits ~100% accuracy
on it, which is *not* a meaningful result on its own. Two good, free,
well-documented options:

| Dataset | Link |
|---|---|
| RAVDESS (recommended, best documented) | https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio |
| TESS | https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess |
| EMO-DB | http://emodb.bilderbar.info/download/ |

To use a real dataset: download it, reorganize the audio files into
```
data/raw_audio/
├── happy/*.wav
├── sad/*.wav
├── angry/*.wav
├── neutral/*.wav
└── fearful/*.wav        (one folder per emotion label)
```
then skip `generate_synthetic_audio.py` and run `extract_features.py`
directly — it works identically on real or synthetic audio.

## Project structure

```
CodeAlpha_EmotionRecognitionSystem/
├── generate_synthetic_audio.py   # creates placeholder .wav files (skip if using a real dataset)
├── extract_features.py           # MFCC + delta feature extraction -> data/features.npz
├── train_model.py                # builds & trains the CNN+LSTM model, evaluates, saves outputs/
├── requirements.txt
├── .gitignore
└── README.md
```

## How to run

```bash
pip install -r requirements.txt

python generate_synthetic_audio.py   # OR: drop a real dataset into data/raw_audio/
python extract_features.py           # extracts MFCC features -> data/features.npz
python train_model.py                # trains the CNN+LSTM model
```

## Model architecture

```
Input (MFCC+delta, time) -> Conv2D(32) -> BN -> MaxPool -> Dropout
                          -> Conv2D(64) -> BN -> MaxPool -> Dropout
                          -> Reshape (to feed LSTM over time)
                          -> LSTM(64) -> Dropout
                          -> Dense(64, relu) -> Dropout
                          -> Dense(n_classes, softmax)
```

## Expected results

On the synthetic demo data, the model reaches ~100% accuracy (the
synthetic signals are acoustically simple and highly separable — this
only validates that the pipeline works end-to-end). On a **real** dataset
like RAVDESS, a model of this size typically lands around **60-75%
accuracy** across 8 emotion classes, or higher (75-90%) if you reduce to
4-5 broader emotion categories — which is a realistic, presentable result.

## Key learnings demonstrated

- Audio signal processing & MFCC feature extraction (`librosa`)
- Building a hybrid CNN+LSTM architecture in TensorFlow/Keras
- Handling variable-length sequences (padding/truncation)
- Multi-class evaluation with macro-averaged metrics
