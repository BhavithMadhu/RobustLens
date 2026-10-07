import torch
from torch.utils.data import Dataset
from .trainer import Trainer

class GroupDROTrainer(Trainer):
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, config=None, num_groups=2, step_size=0.01):
        super().__init__(model, train_loader, val_loader, optimizer, criterion, device, config)
        self.num_groups = num_groups
        self.step_size = step_size
        self.group_weights = torch.ones(num_groups).to(device) / num_groups
        self.criterion_none = torch.nn.CrossEntropyLoss(reduction='none')

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in self.train_loader:
            inputs, targets, group_ids = batch[0], batch[1], batch[2]
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            group_ids = group_ids.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            sample_losses = self.criterion_none(outputs, targets)
            
            group_loss = torch.zeros(self.num_groups).to(self.device)
            group_count = torch.zeros(self.num_groups).to(self.device)
            
            for g in range(self.num_groups):
                idx = (group_ids == g)
                if idx.sum() > 0:
                    group_loss[g] = sample_losses[idx].mean()
                    group_count[g] = idx.sum()
            
            with torch.no_grad():
                self.group_weights = self.group_weights * torch.exp(self.step_size * group_loss.detach())
                self.group_weights = self.group_weights / self.group_weights.sum()
            
            loss = (self.group_weights * group_loss).sum()
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_loss = total_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc


class GroupedColoredMNISTDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        image, label, color = item[0], item[1], item[2]
        
        group_id = 0 if label == color else 1
        
        return image, label, group_id
