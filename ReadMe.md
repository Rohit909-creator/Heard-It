# Heard-IT

**Heard-IT** is a custom wakeword detection system built using a ResNet architecture. Unlike traditional keyword spotting models that rely purely on classification, Heard-IT learns meaningful audio patterns via classification training and later repurposes those learned patterns (embeddings) for more flexible, similarity-based detection.

---

## 🧠 Intuition Behind the Architecture

### Phase 1: Training a Classifier for Embeddings

Initially, the model is trained to classify audio clips of various distinct single-word commands. This forces the ResNet model to learn high-quality internal representations (embeddings) that capture audio nuances such as:

* Phonetic similarities
* Speaker variation
* Pronunciation variability

These embeddings are extracted from the penultimate layer of the model (just before the final fully-connected classifier).

**Why Classification First?**

* Classification provides a structured way to supervise the model into learning audio distinctions.
* Once the model learns to distinguish different words well, the extracted embeddings become valuable for custom wakeword matching.

### Phase 2: Replacing the Classifier with Embedding Output

After training:

* The final classification layer (`model.model.fc[4]`) is replaced with an empty `nn.Sequential()`.
* This lets the model output the embeddings directly instead of classification scores.

These embeddings are then compared to previously recorded wakeword embeddings using a separate module: the **Enhanced Similarity Matcher**.

---

## 📁 Project Directory Structure

```plaintext
├── main.py              # Live wakeword detection using current model
├── Augmentation.py      # Audio data augmentation pipeline
├── Trainer.py           # Model definitions, training loop, and dataloaders
├── Preprocessing.py     # Audio preprocessing and feature extraction
├── Inference.py         # Inference pipeline using trained models
├── Inference2.py        # Extended inference with tabulated comparisons
├── Audio_dir2/          # Training audio dataset (700MB - GDrive linked)
├── Audio_dir3/          # Improved dataset (2GB - GDrive linked)
├── lightning_logs/      # Trained model checkpoints (GDrive linked)
├── record.py            # Script to record 1-second audio clips
├── Audios4testing/      # Test samples recorded using `record.py`
├── mswc3_cache/         # Preprocessed data cache for Audio_dataset3
├── mswc_cache/          # Preprocessed data cache for Audio_dataset2
├── DownstreamTrain.py   # Embedding-based training for similarity matcher
├── ScrapCodes/          # Legacy and trial code (mostly deprecated)
├── Utils.py             # Includes the Enhanced Similarity Matcher and utilities
├── ONNX_CONVERSION.py   # Made for converting pytorch model to onnx
├── ONNX_INFerence.py    # Includes the Enhanced Similarity Matcher and utilities
```

---

## 🔴 Live Detection - `main.py`

### `SimpleMicStream`

* Handles real-time audio stream collection.
* Sample rate: `16000`, Chunk size: `1024`
* Efficiently captures audio in a streaming window (1 second).

### `HotwordDetector`

* Core detection logic.
* Loads model from checkpoint.
* Preprocesses input audio.
* Passes the resulting embeddings to the **Enhanced Similarity Matcher**.
* Matcher must be preloaded with:

  * **Positive embeddings** (i.e., wakeword samples)
  * **Negative embeddings** (i.e., non-wakeword speech/background)

---

## 💡 Enhanced Similarity Matcher (Key Innovation)

The Enhanced Similarity Matcher is what transforms this project into a truly noise-resilient, custom wakeword engine.

### Motivation

The model knows what to detect, but it doesn't inherently know what **not** to detect.
This matcher brings:

* **Negative learning** (using out-of-class samples)
* **Noise sensitivity modeling**
* **Multi-metric robustness**

### 📐 Conceptual View: Embedding Space

* Think of embeddings as points in space.
* Positive wakeword embeddings cluster tightly.
* Negative samples form a contrastive cloud.
* Matcher defines dynamic boundaries that separate wakeword space from everything else.

### 🧮 What It Calculates

Below is a breakdown of how the similarity score is computed:

```python
final_score = (
    weights['cosine'] * cosine_sim +
    weights['avg_pos'] * avg_pos_sim +
    weights['gaussian'] * gaussian_sim -
    weights['negative'] * negative_penalty -
    weights['std'] * std_penalty + 
    boost
)
```

### 🧪 Metrics Used:

* **Cosine similarity**: Measures alignment with the average positive vector.
* **Average cosine to all positives**: Helps account for weak/mumbled wakewords.
* **Gaussian kernel similarity**: Models embedding distance as a soft match (adjusts with noise).
* **Negative penalty**: Penalizes embeddings close to known negatives.
* **Standard deviation check**: Penalizes embeddings that fall too far from known positive distribution.

### 🧊 Noise Robustness:

* If noise level > `0.3`, weights are dynamically rebalanced:

  * Lower reliance on sensitive cosine similarity
  * Increase Gaussian and negative penalties

### 🗣️ Faint Voice Adaptation

* For low-energy speech, penalties are softened
* Boosting logic improves detection when the average positive match still indicates a likely wakeword

### 📊 Dynamic Weights

The matcher adjusts based on:

* Cosine similarity strength
* Gaussian tolerance to noise
* Penalization of standard deviation outliers
* Presence of ambient/negative speech

This adaptive scoring leads to a highly **reliable**, **real-time** wakeword detector even in unpredictable environments.

---

## 🚀 Training Strategy Summary

* Started with standard ResNet classification to learn word-level distinctions
* Used manually curated datasets for high signal-to-noise training
* Switched model to output embeddings
* Applied custom similarity matcher
* Enhanced real-time performance through:

  * Adaptive scoring
  * Noise modeling
  * Negative sample usage

---

## 🔧 Future Directions

* Combine low, mid, and high-level embeddings for richer representations
* Introduce temporal dynamics (sequence matching)
* Use contrastive learning techniques like triplet loss or NT-Xent for embedding refinement
* Build a GUI for easier deployment

---


## 🛠️ Features

- 🎙️ Live audio stream detection (`main.py`)
- 📈 Model training and evaluation with Pytorch Lightning
- 🎧 Preprocessing pipeline using Mel Spectrograms
- 🔁 Data Augmentation support for robust training
- 📂 Google Drive links for datasets and model weights (see below)
- 🔍 Inference module for running predictions on saved checkpoints

---

## 🔗 External Assets

- 🎵 **Audio Dataset (Audio_dir/)**: [Audio_dir](https://drive.google.com/file/d/1nt7fNs_OKq5X4Tk-IXCx8ueU8_-Q-f3I/view?usp=sharing)
- 🧠 **Pretrained Models (lightning_logs/)**: [lightning_logs](https://drive.google.com/drive/folders/1K9Hm2QLoNGrEdXQscS4_aeEHWDdV6Rsd?usp=sharing)
- 🤏🧠 **Pretrained Models (Mini Model 21Mb/)**: [lightning_logs](https://drive.google.com/drive/folders/1K9Hm2QLoNGrEdXQscS4_aeEHWDdV6Rsd?usp=sharing)

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Rohit909-creator/Heard-It
cd Heard-It/src/
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Live Detection

```bash
python main.py
```

### 4. Train a Model

```bash
python Trainer.py
```

### 5. Run Inference

```bash
python Inference2.py
```

---

## 🧪 Sample Use-Case

Detect wakewords like:
- `"Jarvis"`
- `"Heard-it"`
- `"Alexa"`
- Or any **custom phrase** of your choice by using your audio of saying that wakeword as reference!

---

## 💡 Goals & Roadmap

- [x] Real-time Wakeword Detection
- [x] Data Augmentation Pipeline
- [x] Training with ResNet18
- [x] Custom Wakeword Support from User Input
- [ ] Custom Phrase detection Support from User Input
- [ ] Model Quantization for Mobile
- [ ] Android/iOS Compatibility (TFLite/ONNX export)
- [ ] Rich Embedding Visualizer

---