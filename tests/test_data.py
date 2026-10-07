import pytest
import torch
import numpy as np
from src.data.colored_mnist import ColoredMNISTConfig, ColoredMNISTGenerator, get_dataloaders

def test_colored_mnist_config_defaults():
    config = ColoredMNISTConfig()
    assert config.num_classes == 10
    assert config.image_size == 28
    assert config.num_channels == 3
    assert config.train_size > 0
    assert config.test_size > 0

def test_dataset_generation():
    config = ColoredMNISTConfig(train_size=100, test_size=100, seed=42)
    gen = ColoredMNISTGenerator(config)
    data = gen.generate_dataset()
    
    train_images = data['train_images']
    train_labels = data['train_labels']
    assert train_images.shape == (100, 3, 28, 28)
    assert train_images.min() >= 0.0 and train_images.max() <= 1.0
    assert train_labels.min() >= 0 and train_labels.max() <= 9

def test_correlation_strength():
    config_1 = ColoredMNISTConfig(train_size=100, correlation_strength=1.0)
    gen_1 = ColoredMNISTGenerator(config_1)
    data_1 = gen_1.generate_dataset()
    assert data_1['train_images'].shape[0] == 100

    config_0 = ColoredMNISTConfig(train_size=100, correlation_strength=0.0)
    gen_0 = ColoredMNISTGenerator(config_0)
    data_0 = gen_0.generate_dataset()
    assert data_0['train_images'].shape[0] == 100

def test_hard_test_no_matching_colors():
    config = ColoredMNISTConfig(test_size=100)
    gen = ColoredMNISTGenerator(config)
    data = gen.generate_dataset()
    assert data['test_hard_images'].shape == (100, 3, 28, 28)

def test_counterfactual_pairs():
    config = ColoredMNISTConfig(train_size=100)
    gen = ColoredMNISTGenerator(config)
    img1, img2, labels = gen.generate_counterfactual_pairs()
    
    assert img1.shape == img2.shape
    assert (img1 != img2).any()

def test_ood_generation():
    config = ColoredMNISTConfig(test_size=100)
    gen = ColoredMNISTGenerator(config)
    ood_img, ood_lbl = gen.generate_ood_dataset('unseen_colors')
    assert ood_img.shape[0] == 100

def test_dataloader_creation():
    config = ColoredMNISTConfig(train_size=100, test_size=100)
    loaders = get_dataloaders(config, batch_size=16)
    
    images, labels = next(iter(loaders['train']))
    assert images.shape == (16, 3, 28, 28)
