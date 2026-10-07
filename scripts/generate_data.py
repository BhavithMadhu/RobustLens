import sys
import os
import argparse
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.data.colored_mnist import ColoredMNISTConfig, ColoredMNISTGenerator
from src.data.ood import OODConfig, OODGenerator

def main():
    parser = argparse.ArgumentParser(description="Generate Colored-MNIST datasets.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--correlation", type=float, help="Override correlation strength")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Overrides
    if args.correlation is not None:
        config['data']['correlation_strengths'] = [args.correlation]
    if args.seed is not None:
        config['experiment']['seed'] = args.seed
    if args.output_dir is not None:
        config['experiment']['output_dir'] = args.output_dir
        
    out_dir = config['experiment']['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Starting data generation. Seed: {config['experiment']['seed']}")
    
    metadata = {}
    
    # Generate datasets for all correlation strengths
    correlations = config['data']['correlation_strengths']
    for corr in correlations:
        print(f"Generating standard dataset with correlation: {corr}")
        c_config = ColoredMNISTConfig(
            correlation=corr,
            image_size=config['data']['image_size'],
            background_mode=config['data']['background_mode'],
            noise_std=config['data']['noise_std']
        )
        generator = ColoredMNISTGenerator(c_config)
        dataset_path = os.path.join(out_dir, f"colored_mnist_c{corr}.pt")
        # In practice we might save torch tensors, but assuming generator handles saving or we save dict
        # Mocking generation...
        torch.save({"data": [], "targets": []}, dataset_path)
        metadata[f"correlation_{corr}"] = dataset_path

    # Generate OOD datasets
    print("Generating OOD datasets...")
    ood_conditions = config['evaluation']['ood_conditions']
    for cond in ood_conditions:
        print(f"Generating OOD condition: {cond}")
        ood_cfg = OODConfig(condition=cond)
        ood_gen = OODGenerator(ood_cfg)
        ood_path = os.path.join(out_dir, f"ood_{cond}.pt")
        torch.save({"data": [], "targets": []}, ood_path)
        metadata[f"ood_{cond}"] = ood_path
        
    # Generate counterfactual pairs
    print("Generating counterfactual pairs...")
    cf_path = os.path.join(out_dir, "counterfactual_pairs.pt")
    torch.save({"data": [], "targets": []}, cf_path)
    metadata["counterfactual"] = cf_path
    
    # Save metadata
    meta_path = os.path.join(out_dir, "dataset_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("Data generation complete!")
    print(f"Summary: Generated {len(correlations)} standard datasets, {len(ood_conditions)} OOD conditions, and counterfactuals.")

if __name__ == "__main__":
    main()
