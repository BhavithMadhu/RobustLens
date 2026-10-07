import pytest
import torch
from src.utils.metrics import compute_accuracy, compute_confusion_matrix
from src.robustness.pgd_attack import PGDAttack
from src.interpretability.gradcam import GradCAM
from src.interpretability.integrated_gradients import IntegratedGradients

def test_accuracy_computation():
    preds = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    labels = torch.tensor([1, 0])
    acc = compute_accuracy(preds, labels)
    assert acc == 1.0

def test_confusion_matrix_shape():
    preds = torch.randint(0, 10, (100,))
    labels = torch.randint(0, 10, (100,))
    cm = compute_confusion_matrix(preds, labels, num_classes=10)
    assert cm.shape == (10, 10)

def test_adversarial_attack(model):
    device = torch.device("cpu")
    attack = PGDAttack(model, device=device, epsilon=0.1, alpha=0.01, num_steps=10)
    images = torch.rand(4, 3, 28, 28)
    labels = torch.tensor([0, 1, 2, 3])
    
    adv_images = attack.attack(images, labels)
    
    assert adv_images.shape == images.shape
    assert (adv_images != images).any()
    diff = torch.abs(adv_images - images)
    assert diff.max() <= 0.1 + 1e-5

def test_gradcam_output_shape(model):
    target_layer = model.conv3 if hasattr(model, 'conv3') else list(model.children())[0]
    gradcam = GradCAM(model, target_layer=target_layer)
    image = torch.rand(1, 3, 28, 28)
    
    heatmap = gradcam.generate(image, target_class=0)
    gradcam.remove_hooks()
    
    assert heatmap.shape == (28, 28)
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0

def test_integrated_gradients_shape(model):
    device = torch.device("cpu")
    ig = IntegratedGradients(model, device=device)
    image = torch.rand(1, 3, 28, 28)
    
    attributions = ig.compute(image, target_class=0)
    
    assert attributions.shape == image.shape
