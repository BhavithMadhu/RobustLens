import pytest
import torch
import os
import tempfile
from torch.utils.data import TensorDataset, DataLoader
from src.training.trainer import Trainer
from src.training.erm import ERMTrainer
from src.training.gradient_penalty import GradientPenaltyTrainer
from src.training.counterfactual import CounterfactualTrainer
from src.training.group_dro import GroupDROTrainer, GroupedColoredMNISTDataset
from src.models.cheater_cnn import CheaterCNN

def test_erm_training_step(model, sample_batch):
    images, labels = sample_batch
    ds = TensorDataset(images, labels)
    loader = DataLoader(ds, batch_size=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device('cpu')

    trainer = ERMTrainer(model, loader, loader, optimizer, criterion, device)
    metrics_before = trainer.evaluate(loader)
    history = trainer.train(num_epochs=1)
    metrics_after = trainer.evaluate(loader)
    assert len(history['train_loss']) == 1

def test_gradient_penalty_loss(model, sample_batch):
    images, labels = sample_batch
    ds = TensorDataset(images, labels)
    loader = DataLoader(ds, batch_size=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device('cpu')

    trainer = GradientPenaltyTrainer(model, loader, loader, optimizer, criterion, device, lambda_gp=0.5, warmup_epochs=0)
    history = trainer.train(num_epochs=1)
    assert 'train_loss' in history

def test_counterfactual_consistency(model, sample_batch):
    images, labels = sample_batch
    cf_images = torch.rand_like(images)
    
    class PairedDS(torch.utils.data.Dataset):
        def __len__(self): return len(labels)
        def __getitem__(self, idx): return images[idx], cf_images[idx], labels[idx]

    loader = DataLoader(PairedDS(), batch_size=16)
    val_loader = DataLoader(TensorDataset(images, labels), batch_size=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device('cpu')

    trainer = CounterfactualTrainer(model, loader, val_loader, optimizer, criterion, device, lambda_cf=0.5, warmup_epochs=0)
    history = trainer.train(num_epochs=1)
    assert 'train_loss' in history

def test_group_dro_weight_update(model, sample_batch):
    images, labels = sample_batch
    group_ids = torch.randint(0, 2, (len(labels),))

    class GroupDS(torch.utils.data.Dataset):
        def __len__(self): return len(labels)
        def __getitem__(self, idx): return images[idx], labels[idx], group_ids[idx]

    loader = DataLoader(GroupDS(), batch_size=16)
    val_loader = DataLoader(TensorDataset(images, labels), batch_size=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    device = torch.device('cpu')

    trainer = GroupDROTrainer(model, loader, val_loader, optimizer, criterion, device, num_groups=2)
    initial_weights = trainer.group_weights.clone()
    history = trainer.train(num_epochs=1)
    assert not torch.equal(initial_weights, trainer.group_weights)

def test_checkpoint_save_load(model, sample_batch):
    images, labels = sample_batch
    loader = DataLoader(TensorDataset(images, labels), batch_size=16)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device('cpu')
    trainer = Trainer(model, loader, loader, optimizer, criterion, device)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.pt")
        trainer.save_checkpoint(path)
        assert os.path.exists(path)
        
        model_new = CheaterCNN(num_classes=10)
        trainer_new = Trainer(model_new, loader, loader, optimizer, criterion, device)
        trainer_new.load_checkpoint(path)
        
        for p1, p2 in zip(model.parameters(), model_new.parameters()):
            assert torch.equal(p1, p2)
