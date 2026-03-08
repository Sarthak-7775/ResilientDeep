# src/utils/metrics.py
import torch
from sklearn.metrics import accuracy_score, f1_score

def calculate_metrics(all_preds, all_labels):
    """
    Calculates Accuracy and F1-score in O(N) time using vectorized numpy arrays.
    """
    # Convert tensors to numpy arrays
    preds_np = all_preds.cpu().numpy()
    labels_np = all_labels.cpu().numpy()
    
    # Calculate metrics
    acc = accuracy_score(labels_np, preds_np)
    f1 = f1_score(labels_np, preds_np, average='weighted')
    
    return acc, f1