# 505-Sp1786-multiclass-sentiment-analysis
Fine-tuning a BERT model for multiclass sentiment classification using PyTorch, including EDA, evaluation metrics, confusion matrix, and inference pipeline.
# Multiclass Sentiment Analysis using BERT (PyTorch Transfer Learning)

## Overview

This project demonstrates transfer learning by fine-tuning a pretrained BERT model on a multiclass sentiment classification task using PyTorch. The model classifies short text inputs into negative, neutral, or positive sentiment categories.

---

## Dataset

Dataset used: **Sp1786/multiclass-sentiment-analysis-dataset** (Hugging Face)

Each sample contains:

- `text`: Input sentence  
- `label`: Numeric class ID  
- `sentiment`: Human-readable label  

Sentiment classes:

- 0 — Negative  
- 1 — Neutral  
- 2 — Positive  

The dataset was loaded using the Hugging Face `datasets` library with predefined train and test splits.

---

## Exploratory Data Analysis (EDA)

### Dataset Loading

The dataset was loaded using `datasets.load_dataset()`.

### Class Distribution

A bar chart of the training labels was plotted to visualize how samples are distributed across sentiment classes.

### Class Imbalance

The training data shows moderate imbalance, with neutral examples occurring more frequently than positive and negative ones. This imbalance can bias model predictions toward the majority class.

---

## Model Fine-Tuning

A pretrained **bert-base-uncased** model was fine-tuned for sequence classification using PyTorch only.

### Implementation Details

- Framework: PyTorch  
- Tokenizer: BERT tokenizer  
- Model: `BertForSequenceClassification`  
- Output classes: 3  

The pretrained encoder parameters were reused, while the classification layer was trained for the sentiment task.

### Handling Class Imbalance

To reduce bias toward the majority class, class-weighted cross-entropy loss was applied. Class weights were computed from the training labels so that errors on less frequent classes contribute more to the loss during training.

---

## Training Configuration

- Optimizer: AdamW  
- Learning Rate: 2e-5  
- Batch Size: 16  
- Epochs: 3  
- Maximum Sequence Length: 256  
- Learning Rate Scheduler: Linear decay  

Training was performed on GPU when available.

---

## Evaluation Metrics

The fine-tuned model was evaluated on the test set using:

- Accuracy  
- Precision (weighted)  
- Recall (weighted)  
- F1-score (weighted)  

Weighted averaging was used to account for class imbalance.

---

## Confusion Matrix

A confusion matrix was generated to visualize prediction performance across the three sentiment classes and identify common misclassifications.

---

## Inference Pipeline

A function `predict_text(text: str)` was implemented to classify new input text.

The function:

1. Tokenizes the input string  
2. Runs the model in evaluation mode  
3. Computes class probabilities using softmax  
4. Returns the predicted sentiment label and confidence score  

---

## Custom Predictions

The inference function was tested on several custom examples to demonstrate model performance on unseen text inputs.

---

## Results

The model achieved approximately:

- Accuracy ≈ 0.76  
- Precision ≈ 0.76  
- Recall ≈ 0.76  
- F1-score ≈ 0.76 (weighted)  

Performance plateaued after three epochs, indicating convergence for this dataset and configuration.

---

## Conclusion

This project demonstrates the application of transfer learning for text classification using a pretrained transformer model. Despite moderate class imbalance and short input texts, the model learns meaningful sentiment patterns and generalizes to new examples.

---

## Requirements

- Python 3.x  
- PyTorch  
- Transformers  
- Datasets  
- Scikit-learn  
- NumPy  
- Matplotlib  

---

## How to Run

1. Install required libraries  
2. Open the notebook  
3. Run all cells sequentially  
4. Train the model  
5. Evaluate results  
6. Test the inference function with custom text  

---

## Notes

- Fine-tuning was implemented using PyTorch only  
- No high-level Hugging Face training utilities were used  
- All steps (EDA, training, evaluation, inference) are included in the notebook
