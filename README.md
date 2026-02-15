# 505-Sp1786-multiclass-sentiment-analysis
Fine-tuning a BERT model for multiclass sentiment classification using PyTorch, including EDA, evaluation metrics, confusion matrix, and inference pipeline.
# Multiclass Sentiment Analysis using BERT (PyTorch Transfer Learning)

## Overview

This project demonstrates transfer learning by fine-tuning a pretrained **BERT-base** model for multiclass sentiment classification using **PyTorch only**.  
The model predicts whether a given text expresses **negative**, **neutral**, or **positive** sentiment.

All steps required in the assignment — EDA, fine-tuning, evaluation metrics, confusion matrix, and inference pipeline — are implemented without using high-level Hugging Face training utilities.

---

## Dataset

**Dataset:** `Sp1786/multiclass-sentiment-analysis-dataset` (Hugging Face)

Each sample contains:

- `text` – input sentence  
- `label` – numeric class ID  
- `sentiment` – human-readable label  

### Sentiment Classes

| Label | Sentiment |
|-------|-----------|
| 0     | Negative  |
| 1     | Neutral   |
| 2     | Positive  |

**Dataset Size:**
- Training samples: ~31,000  
- Test samples: ~5,000  

The dataset was loaded using the `datasets` library with predefined train/test splits.

---

## Exploratory Data Analysis (EDA)

The class distribution of the training set was visualized using a bar chart.

### Observations
- The dataset shows moderate class imbalance.
- Neutral samples occur more frequently than positive and negative samples.
- Class-weighted loss was used to reduce bias toward the majority class.

---

## Model Architecture

**Base Model:** `bert-base-uncased`

- 12 Transformer encoder layers  
- 768 hidden size  
- 12 attention heads  
- ~110M parameters  

A linear classification head was added for 3-class prediction.

All BERT parameters were fine-tuned end-to-end.

---

## Fine-Tuning Details

**Framework:** PyTorch (no Hugging Face Trainer used)

**Tokenizer:** BERT tokenizer  
**Maximum Sequence Length:** 256  

### Training Configuration

- Optimizer: AdamW  
- Learning Rate: 2e-5  
- Batch Size: 16  
- Epochs: 3  
- Scheduler: Linear decay  
- Loss Function: Class-weighted CrossEntropyLoss  
- Device: GPU (CUDA) when available  

Class weights were computed from training labels to handle imbalance.

---

## Evaluation Metrics

The model was evaluated on the test set using:

- Accuracy  
- Precision (weighted)  
- Recall (weighted)  
- F1-score (weighted)  

### Test Performance

- **Accuracy:** ~0.76  
- **Precision:** ~0.76  
- **Recall:** ~0.76  
- **F1-score (weighted):** ~0.76  

Performance stabilized after three epochs, indicating convergence.

---

## Confusion Matrix

A confusion matrix was generated to analyze misclassifications.

### Key Observations

- Positive and negative classes are well separated.
- Neutral sentiment is the most challenging class.
- Some confusion occurs between neutral and positive samples.

---

## Inference Pipeline

A function `predict_text(text: str)` was implemented to classify new text inputs.

### Pipeline Steps

1. Tokenize input text  
2. Run model in evaluation mode  
3. Apply softmax to obtain probabilities  
4. Return predicted label and confidence score  

### Example Predictions

**Input:** I absolutely loved this product!  
**Prediction:** Positive  
**Confidence:** 0.99  

**Input:** This was the worst experience ever.  
**Prediction:** Negative  
**Confidence:** 0.99  

**Input:** It was okay, nothing special.  
**Prediction:** Neutral  
**Confidence:** 0.69  

---
