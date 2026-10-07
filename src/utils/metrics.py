import torch
import numpy as np
from typing import Dict, List, Any, Union
import json
import csv
import os
from datetime import datetime

def compute_accuracy(model_or_preds, dataloader_or_labels=None, device=None) -> float:
    """
    Compute accuracy. Supports either:
    1) compute_accuracy(model, dataloader, device)
    2) compute_accuracy(preds_tensor, labels_tensor)
    """
    if isinstance(model_or_preds, torch.Tensor) and isinstance(dataloader_or_labels, torch.Tensor):
        preds = model_or_preds
        labels = dataloader_or_labels
        if preds.ndim == 2:
            preds = preds.argmax(dim=1)
        return float((preds == labels).float().mean().item())
    
    model = model_or_preds
    dataloader = dataloader_or_labels
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            if device is not None:
                inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total if total > 0 else 0.0

def compute_confusion_matrix(model_or_preds, dataloader_or_labels=None, device=None, num_classes=10) -> np.ndarray:
    """
    Compute confusion matrix. Supports either:
    1) compute_confusion_matrix(model, dataloader, device, num_classes=10)
    2) compute_confusion_matrix(preds_tensor, labels_tensor, num_classes=10)
    """
    if isinstance(model_or_preds, torch.Tensor) and isinstance(dataloader_or_labels, torch.Tensor):
        preds = model_or_preds
        labels = dataloader_or_labels
        if preds.ndim == 2:
            preds = preds.argmax(dim=1)
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(labels.view(-1), preds.view(-1)):
            cm[t.item(), p.item()] += 1
        return cm

    model = model_or_preds
    dataloader = dataloader_or_labels
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=int)
    with torch.no_grad():
        for inputs, targets in dataloader:
            if device is not None:
                inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            for t, p in zip(targets.view(-1), predicted.view(-1)):
                cm[t.long().item(), p.long().item()] += 1
    return cm

def compute_per_class_accuracy(model, dataloader, device, num_classes=10) -> List[float]:
    cm = compute_confusion_matrix(model, dataloader, device, num_classes)
    accs = []
    for i in range(num_classes):
        total = cm[i].sum()
        acc = cm[i, i] / total if total > 0 else 0.0
        accs.append(acc)
    return accs

class MultiSeedTracker:
    def __init__(self):
        self.results = []
        
    def add_result(self, result: float):
        self.results.append(result)
        
    def get_stats(self) -> Dict[str, float]:
        if not self.results:
            return {'mean': 0.0, 'std': 0.0, 'best': 0.0, 'worst': 0.0}
        arr = np.array(self.results)
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'best': float(np.max(arr)),
            'worst': float(np.min(arr))
        }

class ExperimentLogger:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, 'results.csv')
        self.json_path = os.path.join(log_dir, 'results.json')
        
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'model_name', 'seed', 'correlation_strength', 'metric', 'value'])
                
    def log(self, model_name: str, seed: int, correlation_strength: float, metrics: Dict[str, float]):
        timestamp = datetime.now().isoformat()
        
        # Append to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for k, v in metrics.items():
                writer.writerow([timestamp, model_name, seed, correlation_strength, k, v])
                
        # Append to JSON lines
        log_entry = {
            'timestamp': timestamp,
            'model_name': model_name,
            'seed': seed,
            'correlation_strength': correlation_strength,
            'metrics': metrics
        }
        with open(self.json_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
