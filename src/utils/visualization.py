import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import Dict, List, Any

def setup_plot_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })

def plot_confusion_matrix(cm: np.ndarray, title: str, save_path: str):
    setup_plot_style()
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_accuracy_vs_correlation(results_dict: Dict[float, Dict[str, float]], title: str, save_path: str):
    setup_plot_style()
    plt.figure(figsize=(8, 6))
    
    correlations = sorted(results_dict.keys())
    means = [results_dict[c]['mean'] for c in correlations]
    stds = [results_dict[c]['std'] for c in correlations]
    
    plt.errorbar(correlations, means, yerr=stds, marker='o', capsize=5)
    plt.title(title)
    plt.xlabel('Correlation Strength')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_accuracy_vs_epsilon(results_dict: Dict[float, float], title: str, save_path: str):
    setup_plot_style()
    plt.figure(figsize=(8, 6))
    epsilons = sorted(results_dict.keys())
    accs = [results_dict[e] for e in epsilons]
    plt.plot(epsilons, accs, marker='s', linestyle='-')
    plt.title(title)
    plt.xlabel('Epsilon')
    plt.ylabel('Robust Accuracy')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_multi_seed_comparison(results_dict: Dict[str, Dict[str, float]], title: str, save_path: str):
    setup_plot_style()
    plt.figure(figsize=(10, 6))
    models = list(results_dict.keys())
    means = [results_dict[m]['mean'] for m in models]
    stds = [results_dict[m]['std'] for m in models]
    
    plt.bar(models, means, yerr=stds, capsize=5, color=sns.color_palette("pastel"))
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_ood_radar(results_dict: Dict[str, float], title: str, save_path: str):
    setup_plot_style()
    categories = list(results_dict.keys())
    values = list(results_dict.values())
    
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values += values[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    plt.title(title, y=1.1)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_ablation_bars(results_dict: Dict[str, float], title: str, save_path: str):
    setup_plot_style()
    plt.figure(figsize=(8, 6))
    components = list(results_dict.keys())
    scores = list(results_dict.values())
    
    plt.bar(components, scores, color=sns.color_palette("Set2"))
    plt.title(title)
    plt.ylabel('Performance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_attribution_comparison(images: List[np.ndarray], attrs_list: List[np.ndarray], method_names: List[str], save_path: str):
    setup_plot_style()
    num_methods = len(method_names)
    num_images = len(images)
    
    fig, axes = plt.subplots(num_images, num_methods + 1, figsize=(3*(num_methods+1), 3*num_images))
    if num_images == 1:
        axes = np.expand_dims(axes, 0)
        
    for i in range(num_images):
        axes[i, 0].imshow(np.transpose(images[i], (1, 2, 0)))
        axes[i, 0].axis('off')
        if i == 0: axes[i, 0].set_title('Original')
        
        for j in range(num_methods):
            attr = attrs_list[j][i]
            if len(attr.shape) == 3:
                attr = np.mean(np.abs(attr), axis=0)
            axes[i, j+1].imshow(attr, cmap='hot')
            axes[i, j+1].axis('off')
            if i == 0: axes[i, j+1].set_title(method_names[j])
            
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_training_curves(train_losses: List[float], val_accs: List[float], title: str, save_path: str):
    setup_plot_style()
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color=color)
    ax1.plot(train_losses, color=color, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Validation Accuracy', color=color)
    ax2.plot(val_accs, color=color, label='Val Acc')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(title)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
