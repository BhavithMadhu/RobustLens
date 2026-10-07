"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.

Implements Grad-CAM from scratch using forward and backward hooks,
following Selvaraju et al. (2017). This is a cleaned-up, modular version
inspired by the original Precog project's notebook implementation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GradCAM:
    """Grad-CAM visualization using forward/backward hooks."""

    def __init__(self, model, target_layer):
        """
        Args:
            model: PyTorch model
            target_layer: The convolutional layer to compute Grad-CAM for
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []

        self.hook_handles.append(
            self.target_layer.register_forward_hook(self._forward_hook)
        )
        self.hook_handles.append(
            self.target_layer.register_full_backward_hook(self._backward_hook)
        )

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index. If None, uses predicted class.

        Returns:
            heatmap: numpy array (H, W) normalized to [0, 1]
        """
        self.model.eval()
        self.model.zero_grad()

        input_tensor = input_tensor.clone().requires_grad_(True)
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1

        output.backward(gradient=one_hot)

        # Global average pooling of gradients -> channel importance weights
        alpha_k = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # Weighted combination of activation maps
        heatmap = torch.sum(alpha_k * self.activations, dim=1, keepdim=True)
        heatmap = F.relu(heatmap)

        # Upsample to input size
        heatmap = F.interpolate(
            heatmap, size=input_tensor.shape[2:],
            mode='bilinear', align_corners=False
        )
        heatmap = heatmap.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        heatmap = np.maximum(heatmap, 0)
        hmax = np.max(heatmap)
        if hmax > 0:
            heatmap /= hmax

        return heatmap

    def remove_hooks(self):
        """Remove registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()


def _apply_colormap(heatmap):
    """Apply jet colormap to a heatmap array using matplotlib."""
    colormap = cm.get_cmap('jet')
    colored = colormap(heatmap)[:, :, :3]  # Drop alpha channel
    return colored.astype(np.float32)


def visualize_gradcam(model, image, target_layer, target_class, save_path):
    """
    Generate and save a Grad-CAM visualization.

    Args:
        model: PyTorch model
        image: Input tensor (1, C, H, W)
        target_layer: Target convolutional layer module
        target_class: Class to visualize (int)
        save_path: Path to save the figure
    """
    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam.generate(image, target_class)
    gradcam.remove_hooks()

    img_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)
    if img_np.shape[2] == 1:
        img_np = np.repeat(img_np, 3, axis=2)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

    heatmap_colored = _apply_colormap(heatmap)

    cam_result = 0.5 * heatmap_colored + 0.5 * img_np
    cam_result = cam_result / (np.max(cam_result) + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis('off')

    axes[2].imshow(cam_result)
    axes[2].set_title("Overlay")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def compare_gradcam(models_dict, image, target_layer_name, save_path):
    """
    Compare Grad-CAM across multiple models.

    Args:
        models_dict: Dict mapping model_name -> model
        image: Input tensor (1, C, H, W)
        target_layer_name: Name of target layer (e.g., 'features.4')
        save_path: Path to save the comparison figure
    """
    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(4 * (n_models + 1), 4))

    img_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)
    if img_np.shape[2] == 1:
        img_np = np.repeat(img_np, 3, axis=2)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

    axes[0].imshow(img_np)
    axes[0].set_title("Original")
    axes[0].axis('off')

    for idx, (name, model) in enumerate(models_dict.items()):
        target_layer = None
        for n, m in model.named_modules():
            if n == target_layer_name:
                target_layer = m
                break

        if target_layer is None:
            axes[idx + 1].text(0.5, 0.5, f"Layer not found",
                             ha='center', va='center')
            axes[idx + 1].set_title(name)
            axes[idx + 1].axis('off')
            continue

        gradcam = GradCAM(model, target_layer)
        heatmap = gradcam.generate(image)
        gradcam.remove_hooks()

        heatmap_colored = _apply_colormap(heatmap)
        cam_result = 0.5 * heatmap_colored + 0.5 * img_np
        cam_result = cam_result / (np.max(cam_result) + 1e-8)

        axes[idx + 1].imshow(cam_result)
        axes[idx + 1].set_title(name)
        axes[idx + 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
