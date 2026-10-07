import torch
import torch.nn as nn
import pandas as pd

class PGDAttack:
    def __init__(self, model, device, epsilon, alpha, num_steps, targeted=True):
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.targeted = targeted
        self.criterion = nn.CrossEntropyLoss()

    def attack(self, images, labels, target_labels=None):
        images = images.clone().detach().to(self.device)
        labels = labels.to(self.device)
        
        if self.targeted and target_labels is not None:
            targets = target_labels.to(self.device)
        else:
            targets = labels
            
        adv_images = images.clone().detach()
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.epsilon, self.epsilon)
        adv_images = torch.clamp(adv_images, 0, 1).detach()
        
        for _ in range(self.num_steps):
            adv_images.requires_grad = True
            
            outputs = self.model(adv_images)
            loss = self.criterion(outputs, targets)
            
            self.model.zero_grad()
            loss.backward()
            
            grad_sign = adv_images.grad.sign()
            
            if self.targeted:
                adv_images = adv_images.detach() - self.alpha * grad_sign
            else:
                adv_images = adv_images.detach() + self.alpha * grad_sign
                
            eta = torch.clamp(adv_images - images, min=-self.epsilon, max=self.epsilon)
            adv_images = torch.clamp(images + eta, min=0, max=1).detach()
            
        return adv_images

class AdversarialEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def evaluate_robustness(self, dataloader, epsilon_values, num_steps=20, alpha=None):
        results = {}
        for eps in epsilon_values:
            a = alpha if alpha is not None else eps / 4
            attack = PGDAttack(self.model, self.device, eps, a, num_steps, targeted=False)
            
            correct_clean = 0
            correct_adv = 0
            total = 0
            
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(images)
                    preds_clean = outputs.argmax(dim=1)
                    correct_clean += (preds_clean == labels).sum().item()
                    
                adv_images = attack.attack(images, labels)
                with torch.no_grad():
                    adv_outputs = self.model(adv_images)
                    preds_adv = adv_outputs.argmax(dim=1)
                    correct_adv += (preds_adv == labels).sum().item()
                    
                total += labels.size(0)
                
            clean_acc = correct_clean / total if total > 0 else 0
            adv_acc = correct_adv / total if total > 0 else 0
            
            results[eps] = {
                'clean_accuracy': clean_acc,
                'adversarial_accuracy': adv_acc,
                'accuracy_degradation': clean_acc - adv_acc,
                'attack_success_rate': 1.0 - (adv_acc / clean_acc) if clean_acc > 0 else 0
            }
            
        return results

    def evaluate_targeted(self, dataloader, source_class, target_class, epsilon, num_steps):
        alpha = epsilon / 4
        attack = PGDAttack(self.model, self.device, epsilon, alpha, num_steps, targeted=True)
        
        success = 0
        total = 0
        
        for images, labels in dataloader:
            mask = labels == source_class
            if not mask.any():
                continue
                
            filtered_images = images[mask].to(self.device)
            filtered_labels = labels[mask].to(self.device)
            targets = torch.full_like(filtered_labels, target_class).to(self.device)
            
            adv_images = attack.attack(filtered_images, filtered_labels, targets)
            
            with torch.no_grad():
                adv_outputs = self.model(adv_images)
                preds = adv_outputs.argmax(dim=1)
                success += (preds == targets).sum().item()
                total += targets.size(0)
                
        return {
            'targeted_attack_success_rate': success / total if total > 0 else 0.0,
            'source_class': source_class,
            'target_class': target_class
        }

    def sweep_epsilon(self, dataloader, epsilon_values):
        results = self.evaluate_robustness(dataloader, epsilon_values)
        records = []
        for eps, metrics in results.items():
            record = {'epsilon': eps}
            record.update(metrics)
            records.append(record)
        return pd.DataFrame(records)
