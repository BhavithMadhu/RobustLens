from enum import Enum
import torch
import torch.nn as nn
import pandas as pd

class DebiasingMethod(Enum):
    ERM = "ERM"
    GRADIENT_PENALTY = "GRADIENT_PENALTY"
    COUNTERFACTUAL = "COUNTERFACTUAL"
    GROUP_DRO = "GROUP_DRO"

class BaseTrainer:
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 1e-3))
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

class ERMTrainer(BaseTrainer):
    pass

class GradientPenaltyTrainer(BaseTrainer):
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        penalty_weight = self.config.get('penalty_weight', 0.1)
        
        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device).requires_grad_(True)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            gradients = torch.autograd.grad(outputs=loss, inputs=inputs,
                                            create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients.view(gradients.size(0), -1)
            gradient_penalty = penalty_weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            
            total_loss_batch = loss + gradient_penalty
            total_loss_batch.backward()
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            
        return total_loss / len(self.train_loader)

class GroupDROTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, device, config):
        super().__init__(model, train_loader, val_loader, device, config)
        self.n_groups = config.get('n_groups', 4)
        self.group_weights = torch.ones(self.n_groups).to(device)
        self.step_size = config.get('step_size', 0.01)
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            if len(batch) == 3:
                inputs, targets, groups = batch
            else:
                inputs, targets = batch
                groups = torch.zeros_like(targets)
                
            inputs, targets, groups = inputs.to(self.device), targets.to(self.device), groups.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            losses = torch.zeros(self.n_groups).to(self.device)
            for i in range(self.n_groups):
                mask = groups == i
                if mask.any():
                    losses[i] = self.criterion(outputs[mask], targets[mask])
                    
            self.group_weights = self.group_weights * torch.exp(self.step_size * losses.detach())
            self.group_weights = self.group_weights / self.group_weights.sum()
            
            loss = (self.group_weights * losses).sum()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

class CounterfactualTrainer(BaseTrainer):
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            if len(batch) >= 3:
                inputs, targets, cf_inputs = batch[0], batch[1], batch[2]
            else:
                inputs, targets = batch
                cf_inputs = inputs # Fallback if no counterfactuals
                
            inputs, targets, cf_inputs = inputs.to(self.device), targets.to(self.device), cf_inputs.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            cf_outputs = self.model(cf_inputs)
            
            loss_standard = self.criterion(outputs, targets)
            loss_cf = self.criterion(cf_outputs, targets)
            
            loss = loss_standard + self.config.get('cf_weight', 1.0) * loss_cf
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

def get_trainer(method, model, train_loader, val_loader, device, config):
    if method == DebiasingMethod.ERM:
        return ERMTrainer(model, train_loader, val_loader, device, config)
    elif method == DebiasingMethod.GRADIENT_PENALTY:
        return GradientPenaltyTrainer(model, train_loader, val_loader, device, config)
    elif method == DebiasingMethod.GROUP_DRO:
        return GroupDROTrainer(model, train_loader, val_loader, device, config)
    elif method == DebiasingMethod.COUNTERFACTUAL:
        return CounterfactualTrainer(model, train_loader, val_loader, device, config)
    else:
        raise ValueError(f"Unknown method {method}")

def compare_methods(methods_list, train_loader, val_loader, test_loader, hard_test_loader, device, config):
    records = []
    
    for method in methods_list:
        model = config['model_builder']().to(device)
        trainer = get_trainer(method, model, train_loader, val_loader, device, config)
        
        epochs = config.get('epochs', 5)
        for epoch in range(epochs):
            trainer.train_epoch()
            
        model.eval()
        from src.evaluation.evaluator import ModelEvaluator
        evaluator = ModelEvaluator(model, device)
        
        test_res = evaluator.evaluate_standard(test_loader)
        hard_res = evaluator.evaluate_hard(hard_test_loader)
        
        records.append({
            'Method': method.value,
            'Test_Accuracy': test_res['accuracy'],
            'Hard_Test_Accuracy': hard_res['accuracy']
        })
        
    return pd.DataFrame(records)
