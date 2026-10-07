import torch
import torch.nn as nn
import json
import numpy as np

class ModelEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def evaluate_standard(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        class_correct = {}
        class_total = {}
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                for t, p in zip(targets.cpu().numpy(), predicted.cpu().numpy()):
                    class_correct[t] = class_correct.get(t, 0) + (1 if t == p else 0)
                    class_total[t] = class_total.get(t, 0) + 1
                    
        per_class_accuracy = {k: class_correct[k] / class_total[k] for k in class_correct}
        
        return {
            'accuracy': correct / total,
            'loss': total_loss / total,
            'per_class_accuracy': per_class_accuracy
        }

    def evaluate_hard(self, hard_dataloader):
        return self.evaluate_standard(hard_dataloader)

    def evaluate_ood(self, ood_dataloaders_dict):
        results = {}
        for condition_name, loader in ood_dataloaders_dict.items():
            metrics = self.evaluate_standard(loader)
            results[condition_name] = metrics['accuracy']
        return results

    def evaluate_all(self, dataloaders_dict):
        results = {}
        if 'standard' in dataloaders_dict:
            results['standard'] = self.evaluate_standard(dataloaders_dict['standard'])
        if 'hard' in dataloaders_dict:
            results['hard'] = self.evaluate_hard(dataloaders_dict['hard'])
        
        ood_loaders = {k: v for k, v in dataloaders_dict.items() if k not in ['standard', 'hard']}
        if ood_loaders:
            results['ood'] = self.evaluate_ood(ood_loaders)
            results['ood_robustness_score'] = self.compute_ood_robustness_score(results['ood'])
            
        return results

    def compute_ood_robustness_score(self, ood_results):
        if not ood_results:
            return 0.0
        return sum(ood_results.values()) / len(ood_results)

    def save_results(self, results, path):
        # Convert any non-serializable objects
        def convert(o):
            if isinstance(o, np.int64): return int(o)
            if isinstance(o, np.float32): return float(o)
            raise TypeError
        with open(path, 'w') as f:
            json.dump(results, f, indent=4, default=convert)
