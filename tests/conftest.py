import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.data.colored_mnist import ColoredMNISTConfig, ColoredMNISTGenerator, ColoredMNISTDataset, get_dataloaders
from src.models.cheater_cnn import CheaterCNN

@pytest.fixture
def device():
    return torch.device("cpu")

@pytest.fixture
def small_dataset():
    config = ColoredMNISTConfig(train_size=100, test_size=100, correlation_strength=0.9, seed=42)
    # Generate mock data that ColoredMNISTDataset would typically use
    images = torch.rand(100, 3, 28, 28)
    labels = torch.randint(0, 10, (100,))
    colors = torch.randint(0, 10, (100,))
    return ColoredMNISTDataset(images, labels, colors)

@pytest.fixture
def model():
    return CheaterCNN(num_classes=10)

@pytest.fixture
def trained_model(model):
    # Mock trained model
    return model

@pytest.fixture
def sample_batch():
    images = torch.rand(16, 3, 28, 28)
    labels = torch.randint(0, 10, (16,))
    return images, labels
