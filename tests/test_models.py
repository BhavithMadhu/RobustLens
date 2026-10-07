import pytest
import torch
from src.models.cheater_cnn import CheaterCNN
from src.models.robust_cnn import RobustCNN

def test_cheater_cnn_forward():
    model = CheaterCNN(num_classes=10)
    x = torch.rand(1, 3, 28, 28)
    out = model(x)
    assert out.shape == (1, 10)

def test_cheater_cnn_parameters():
    model = CheaterCNN(num_classes=10)
    num_params = sum(p.numel() for p in model.parameters())
    assert 10000 < num_params < 20000

def test_robust_cnn_forward():
    model = RobustCNN(num_classes=10)
    x = torch.rand(1, 3, 28, 28)
    out = model(x)
    assert out.shape == (1, 10)

def test_robust_cnn_with_batchnorm():
    model = RobustCNN(num_classes=10, batch_norm=True)
    has_bn = any(isinstance(module, torch.nn.BatchNorm2d) for module in model.modules())
    assert has_bn

def test_feature_maps():
    model = CheaterCNN(num_classes=10)
    x = torch.rand(1, 3, 28, 28)
    features = model.get_feature_maps(x, layer_idx=0)
    assert features is not None
    assert len(features.shape) == 4  # (B, C, H, W)

def test_model_gradient_flow():
    model = CheaterCNN(num_classes=10)
    x = torch.rand(1, 3, 28, 28)
    y = torch.tensor([0])
    
    criterion = torch.nn.CrossEntropyLoss()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    
    for name, param in model.named_parameters():
        assert param.grad is not None
        assert param.grad.sum() != 0
