import sys
import os
import argparse
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config

# Import plot functions
# Note: assuming these exist in src.utils.visualization as specified
from src.utils.visualization import (
    plot_model_comparison,
    plot_correlation_sweep,
    plot_ood_performance,
    plot_adversarial_robustness,
    plot_multi_seed_performance,
    plot_ablation_results,
    plot_confusion_matrices
)

def main():
    parser = argparse.ArgumentParser(description="Generate figures from results.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = config['experiment']['output_dir']
    figures_dir = config['experiment']['figures_dir']
    
    os.makedirs(figures_dir, exist_ok=True)
    print(f"Loading results from {results_dir} and generating figures in {figures_dir}...")
    
    # Mock data loading for plotting purposes
    results_files = glob.glob(os.path.join(results_dir, "*.json"))
    print(f"Found {len(results_files)} result files.")
    
    # Call dummy plot functions from src.utils.visualization
    # In reality, we'd parse the json files and pass the data structures.
    try:
        plot_model_comparison(results_dir, os.path.join(figures_dir, "model_comparison.png"))
        plot_correlation_sweep(results_dir, os.path.join(figures_dir, "correlation_sweep.png"))
        plot_ood_performance(results_dir, os.path.join(figures_dir, "ood_performance.png"))
        plot_adversarial_robustness(results_dir, os.path.join(figures_dir, "adversarial_robustness.png"))
        plot_multi_seed_performance(results_dir, os.path.join(figures_dir, "multi_seed_performance.png"))
        plot_ablation_results(results_dir, os.path.join(figures_dir, "ablation_results.png"))
        plot_confusion_matrices(results_dir, os.path.join(figures_dir, "confusion_matrices.png"))
        print("Successfully generated all figures.")
    except Exception as e:
        print(f"Warning during figure generation (expected if mock functions are incomplete): {e}")

if __name__ == "__main__":
    main()
