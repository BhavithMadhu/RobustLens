"""
Integrated Gradients attribution method.

Implements Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks".
This is a NEW method not present in the original Precog project, which only
used Grad-CAM.
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


class IntegratedGradients:
    """Integrated Gradients attribution method."""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def compute(self, input_tensor, target_class, baseline=None, num_steps=50):
        """
        Compute Integrated Gradients attribution.

        IG(x) = (x - x') * integral_0^1 (dF/dx at x' + alpha*(x-x')) d_alpha

        Args:
            input_tensor: Input image (1, C, H, W)
            target_class: Target class index
            baseline: Baseline image. Defaults to zero tensor.
            num_steps: Number of interpolation steps for Riemann sum

        Returns:
            attribution: Tensor of same shape as input
        """
        if baseline is None:
            baseline = torch.zeros_like(input_tensor).to(self.device)

        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        baseline = baseline.to(self.device)

        # Create interpolated inputs along the path from baseline to input
        alphas = torch.linspace(0, 1, num_steps).to(self.device)
        delta = input_tensor - baseline

        # Process in smaller batches to avoid memory issues
        batch_size = min(num_steps, 25)
        all_gradients = []

        for start in range(0, num_steps, batch_size):
            end = min(start + batch_size, num_steps)
            batch_alphas = alphas[start:end]
            path_inputs = baseline + batch_alphas.view(-1, 1, 1, 1) * delta
            path_inputs = path_inputs.detach().requires_grad_(True)

            outputs = self.model(path_inputs)
            target_outputs = outputs[:, target_class]

            gradients = torch.autograd.grad(
                outputs=target_outputs,
                inputs=path_inputs,
                grad_outputs=torch.ones_like(target_outputs),
                create_graph=False
            )[0]

            all_gradients.append(gradients.detach())

        all_gradients = torch.cat(all_gradients, dim=0)

        # Average gradients (Riemann sum approximation of the integral)
        avg_gradients = torch.mean(all_gradients, dim=0, keepdim=True)

        # Scale by (input - baseline)
        integrated_gradients = delta * avg_gradients

        return integrated_gradients


def visualize_integrated_gradients(model, image, target_class, save_path):
    """Generate and save an Integrated Gradients visualization."""
    ig = IntegratedGradients(model, image.device)
    attribution = ig.compute(image, target_class)

    attr_np = attribution.squeeze().cpu().detach().numpy()
    if attr_np.ndim == 3:
        attr_np = np.sum(np.abs(attr_np), axis=0)

    vmax = np.max(np.abs(attr_np)) + 1e-8

    img_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    im = axes[1].imshow(attr_np, cmap='hot', vmin=0, vmax=vmax)
    axes[1].set_title("Integrated Gradients")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlay: attribution magnitude on image
    attr_normalized = attr_np / vmax
    overlay = 0.6 * img_np + 0.4 * plt.cm.hot(attr_normalized)[:, :, :3]
    overlay = np.clip(overlay, 0, 1)
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def compare_attributions(model, image, target_class, methods_results, save_path):
    """
    Compare multiple attribution methods side-by-side.

    Args:
        model: PyTorch model
        image: Input tensor (1, C, H, W)
        target_class: Target class
        methods_results: Dict mapping method_name -> attribution array
        save_path: Path to save figure
    """
    n_methods = len(methods_results)
    fig, axes = plt.subplots(1, n_methods + 1, figsize=(4 * (n_methods + 1), 4))

    img_np = image.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    if img_np.shape[2] == 1:
        img_np = np.repeat(img_np, 3, axis=2)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

    axes[0].imshow(img_np)
    axes[0].set_title("Original")
    axes[0].axis('off')

    for idx, (name, attr) in enumerate(methods_results.items()):
        if isinstance(attr, np.ndarray) and attr.ndim == 2:
            # Already a 2D heatmap (e.g., Grad-CAM)
            colormap = cm.get_cmap('jet')
            heatmap_colored = colormap(attr)[:, :, :3]
            cam_result = 0.5 * heatmap_colored + 0.5 * img_np
            cam_result = cam_result / (np.max(cam_result) + 1e-8)
            axes[idx + 1].imshow(cam_result)
        else:
            # Tensor attribution (e.g., Integrated Gradients)
            if isinstance(attr, torch.Tensor):
                attr_np = attr.squeeze().cpu().detach().numpy()
            else:
                attr_np = attr
            if attr_np.ndim == 3:
                attr_np = np.sum(np.abs(attr_np), axis=0)
            vmax = np.max(np.abs(attr_np)) + 1e-8
            axes[idx + 1].imshow(attr_np, cmap='hot', vmin=0, vmax=vmax)

        axes[idx + 1].set_title(name)
        axes[idx + 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
