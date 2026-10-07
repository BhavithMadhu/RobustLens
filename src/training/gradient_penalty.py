import torch
from .trainer import Trainer

class GradientPenaltyTrainer(Trainer):
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, config=None, lambda_gp=0.5, warmup_epochs=5):
        super().__init__(model, train_loader, val_loader, optimizer, criterion, device, config)
        self.lambda_gp = lambda_gp
        self.warmup_epochs = warmup_epochs

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs.requires_grad = True
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            ce_loss = self.criterion(outputs, targets)
            
            batch_indices = torch.arange(inputs.size(0)).to(self.device)
            grad_outputs = torch.ones(inputs.size(0)).to(self.device)
            true_class_logits = outputs[batch_indices, targets]
            
            gradients = torch.autograd.grad(
                outputs=true_class_logits,
                inputs=inputs,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            
            grad_norm_sq = (gradients.view(gradients.size(0), -1).norm(2, dim=1) ** 2).mean()
            loss = ce_loss + self.lambda_gp * grad_norm_sq
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_loss = total_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
