import sys
import os
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.metrics import ExperimentLogger
from src.data.colored_mnist import get_dataloaders
from src.models.base import get_model
from src.debiasing.methods import get_trainer

def main():
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--method", type=str, default="erm")
    parser.add_argument("--correlation", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    config = load_config(args.config)
    
    method = args.method
    corr = args.correlation if args.correlation is not None else config['data']['default_correlation']
    seed = args.seed if args.seed is not None else config['experiment']['seed']
    epochs = args.epochs if args.epochs is not None else config['training']['epochs']
    out_dir = args.output_dir if args.output_dir is not None else config['experiment']['output_dir']
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Init logger
    logger = ExperimentLogger(out_dir)
    
    print(f"Training using method: {method} | Correlation: {corr} | Seed: {seed} | Epochs: {epochs}")
    
    # Load data
    print("Loading data...")
    train_loader, val_loader = get_dataloaders(config, corr, seed)
    
    # Load model
    print("Initializing model...")
    model = get_model(config['model']['name'], config['model'])
    
    # Get trainer
    print("Initializing trainer...")
    trainer = get_trainer(method, model, config)
    
    # Train
    print("Starting training...")
    history = trainer.train(train_loader, val_loader, epochs)
    
    # Save model and metrics
    model_path = os.path.join(out_dir, f"model_{method}_c{corr}_s{seed}.pth")
    torch.save(model.state_dict(), model_path)
    logger.log_metrics(history, f"train_metrics_{method}_c{corr}_s{seed}.json")
    
    print("Training complete! Model saved to:", model_path)

if __name__ == "__main__":
    main()
