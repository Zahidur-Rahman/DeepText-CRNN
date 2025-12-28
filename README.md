# DeepText-CRNN: Scene Text Recognition with PyTorch

> A robust Optical Character Recognition (OCR) system implementing the CRNN (Convolutional Recurrent Neural Network) architecture for accurate sequence-based text recognition.

## 📌 Project Overview
**DeepText-CRNN** is a deep learning model designed to recognize text from images (scene text, handwriting, or synthetic text). It combines the power of **Convolutional Neural Networks (CNN)** for visual feature extraction with **Recurrent Neural Networks (RNN)** for sequence modeling, trained end-to-end using **Connectionist Temporal Classification (CTC) Loss**.

This architecture allows the model to learn from data without requiring explicit character-level alignment, making it highly effective for variable-length text recognition tasks.

## 🚀 Key Features
- **Hybrid Architecture**: Integrates **VGG-style CNN** for feature extraction and **Bidirectional LSTM (BiLSTM)** for context-aware sequence learning.
- **CTC Loss**: Utilizes Connectionist Temporal Classification for alignment-free sequence training.
- **End-to-End Pipeline**: Includes complete scripts for data preparation, training, validation, and inference.
- **Modular Design**: Clean separation of configuration, model definition, and data loading logic.
- **Custom Dataset Support**: Designed to work with MJSynth (Synth90k) and can be adapted for other datasets.

## 🛠️ Technical Architecture
The model relies on three key components:
1.  **Convolutional Layers**: A 7-layer CNN backbone (VGG-variant) extracts a feature sequence from the input image.
2.  **Recurrent Layers**: A deep Bidirectional LSTM network propagates information through the sequence, capturing long-range dependencies.
3.  **Transcription Layer**: A fully connected layer followed by Softmax and CTC decoding converts the per-frame predictions into the final text label.

## 📂 Project Structure
```bash
├── checkpoints/       # Saved model weights
├── config.py          # Hyperparameters and file path configurations
├── dataset.py         # Custom pytorch Dataset and DataLoader
├── evaluate.py        # Model evaluation scripts
├── inference.py       # Script for running predictions on single images
├── model.py           # CRNN model architecture definition
├── train.py           # Main training loop with validation and checkpointing
├── utils.py           # Helper functions
└── ...
```

## ⚡ Getting Started

### Prerequisites
- Python 3.x
- PyTorch
- OpenCV
- Numpy

### Installation
Clone the repository and install dependencies:
```bash
pip install torch torchvision opencv-python numpy tqdm
```

### Configuration
Edit `config.py` to set your dataset paths and hyperparameters:
```python
# config.py
DATA_DIR = "/path/to/your/dataset"
BATCH_SIZE = 512
LEARNING_RATE = 0.0005
```

### Training
To start training the model:
```bash
python train.py
```
*The script automatically handles validation and saves the best model to the `checkpoints/` directory.*

### Inference
To recognize text from a single image:
```bash
python inference.py --image path/to/image.jpg
```

## 📊 Performance
The model is optimized for the MJSynth dataset and achieves high accuracy on standard benchmarks after ~10 epochs of training.

## 📜 License
This project is open-source and available for educational and research purposes.
