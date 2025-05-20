# 🔊 Heard-it: Advanced Wakeword & Phrase Detection

**Heard-it** is an advanced audio-based wakeword and phrase detection engine designed to not just detect specific hotwords, but also understand **commands** and **phrases**. Built with deep learning at its core, this framework is designed for real-time, on-device audio inference and training.

> 🚧 *This project is a work in progress – with exciting features in development, including customizable classification from rich audio embeddings. Although there are trial versions of this available here*

---

## 📦 Project Structure

```
├── main.py              # Live Wakeword Detection using current model
├── Augmentation.py      # Audio data augmentation pipeline
├── Trainer.py           # Model definitions, training loop, and dataloaders
├── Preprocessing.py     # Audio preprocessing and feature extraction
├── Inference.py         # Inference pipeline using trained models
├── Audio_dir/           # Directory containing training audio (linked via GDrive)
├── lightning_logs/      # Trained model checkpoints (linked via GDrive)
```

---

## 🧠 Model Architectures

We’re currently experimenting with the following architectures:

- ✅ **ResNet-18** – Best performing so far in terms of accuracy and generalization.
- 🧪 **CRNN (CNN + RNN)** – Tested, but hasn’t shown promising results yet.
- 🔜 **(Planned)** Transformer-based audio models or wav2vec-inspired encoders.

Our goal is to create **rich embeddings** that allow for **custom wakeword** and **phrase classification** from user-supplied audio.

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

## 👨‍💻 Contributors

- **Me** – [@Rohit Francis](https://github.com/Rohit909-creator)