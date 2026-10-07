import torch
from .trainer import Trainer

class CounterfactualTrainer(Trainer):
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, config=None, lambda_cf=0.5, warmup_epochs=2):
        super().__init__(model, train_loader, val_loader, optimizer, criterion, device, config)
        self.lambda_cf = lambda_cf
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in self.train_loader:
            inputs, cf_inputs, targets = batch[0], batch[1], batch[2]
            inputs, cf_inputs, targets = inputs.to(self.device), cf_inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            ce_loss = self.criterion(outputs, targets)
            
            if self.current_epoch >= self.warmup_epochs:
                cf_outputs = self.model(cf_inputs)
                cf_loss = torch.mean((outputs - cf_outputs) ** 2)
                loss = ce_loss + self.lambda_cf * cf_loss
            else:
                loss = ce_loss
                
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_loss = total_loss / total
        epoch_acc = correct / total
        self.current_epoch += 1
        return epoch_loss, epoch_acc
