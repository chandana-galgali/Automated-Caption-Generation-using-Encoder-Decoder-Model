# 💎 Automated-Caption-Generation-using-Encoder-Decoder-Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📖 Overview
This project implements a dual-purpose computer vision system designed to analyze jewelry images. It utilizes **Transfer Learning** with a pre-trained VGG-16 architecture to perform two distinct tasks:
1.  **Classification:** Accurately identifying whether an image contains a *Necklace* or *Earrings*.
2.  **Image Captioning:** Generating natural language descriptions (captions) for the jewelry items.

The model achieves high accuracy (~95% on validation) by combining Convolutional Neural Networks (CNNs) for feature extraction with Gated Recurrent Units (GRUs) for sequence processing.

## 📂 Dataset
The dataset was constructed by aggregating and augmenting data from open-source repositories on Hugging Face.

* **Sources:** `AI Tool Pool Jewelry Vision` (Primary) and `imnaveenk/necklace` (Supplementary).
* **Augmentation:** To resolve class imbalance and limited data, we applied a rigorous augmentation pipeline (flips, rotations, brightness adjustments), expanding the dataset by **6x**.
* **Final Size:** ~31,000 images.
* **Preprocessing:**
    * Resizing to **224x224** pixels.
    * Normalization using ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

## 🏗️ Model Architecture

### 1. Classification Module
Instead of a standard CNN classifier, we employed a hybrid approach:
* **Encoder:** Frozen **VGG-16** backbone to extract high-level spatial feature maps.
* **Classifier:** The feature maps are flattened into a sequence and passed through a **GRU (Gated Recurrent Unit)**. This allows the model to capture spatial dependencies as sequential information before making the final class prediction.

### 2. Captioning Module (Encoder-Decoder)
For generating descriptions, we used a standard Encoder-Decoder architecture:
* **Encoder:** **VGG-16** (Pre-trained) extracts a fixed-length feature vector from the image.
* **Decoder:** A **GRU** language model initializes with the image features and predicts the caption word-by-word using a built-in vocabulary.

## 🛠️ Tech Stack
* **Core Framework:** PyTorch, Torchvision
* **Data Handling:** Pandas, NumPy
* **Image Processing:** PIL (Python Imaging Library)
* **NLP:** NLTK (for tokenization and vocabulary building)
* **Visualization:** Matplotlib, IPyWidgets (for the interactive UI)

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch torchvision pandas matplotlib nltk ipywidgets
