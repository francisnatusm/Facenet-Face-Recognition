# 👤 FaceNet Face Recognition System

A complete implementation of **FaceNet** for face verification and recognition using triplet loss and 128-dimensional face embeddings.

## 📋 Overview

FaceNet is a state-of-the-art face recognition system that:
- Maps faces to 128-dimensional embedding vectors
- Uses triplet loss to ensure same-person embeddings are close
- Different-person embeddings are far apart
- Enables both face verification (is this person who they claim?) and face recognition (who is this person?)

## 🎯 What This Project Does

- ✅ **Face Verification** - Confirm if a person matches their claimed identity
- ✅ **Face Recognition** - Identify unknown faces from a database
- ✅ **Triplet Loss** - Custom loss function for learning face embeddings
- ✅ **Pre-trained Model** - Uses Inception-based FaceNet architecture
- ✅ **Visual Feedback** - Displays images with verification results

## 🧠 How FaceNet Works

Input Face (160x160x3)
↓
Deep CNN (Inception)
↓
128-dimensional Embedding
↓
L2 Normalization
↓
Distance Comparison
↓
Verification/Recognition


**Triplet Loss Formula:**

Loss = max(||A-P||² - ||A-N||² + α, 0)

- A = Anchor (reference face)
- P = Positive (same person as anchor)
- N = Negative (different person)
- α = Margin (0.2)

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- TensorFlow 2.x

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/facenet-recognition.git
cd facenet-recognition

pip install -r requirements.txt