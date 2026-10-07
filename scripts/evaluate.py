import sys
import os
import argparse
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.models.base import get_model
from src.evaluation.evaluator import ModelEvaluator
from src.robustness.pgd_attack import AdversarialEvaluator

def main():
    parser = argparse.ArgumentParser(description="Evaluate model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--eval_type", type=str, choices=['standard', 'hard', 'ood', 'adversarial', 'all'], default='all')
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = config['experiment']['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Evaluating model: {args.model_path}")
    print(f"Evaluation type: {args.eval_type}")
    
    # Load model
    model = get_model(config['model']['name'], config['model'])
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()
    
    evaluator = ModelEvaluator(model, config)
    adv_evaluator = AdversarialEvaluator(model, config)
    
    results = {}
    
    if args.eval_type in ['standard', 'all']:
        print("Running standard evaluation...")
        results['standard'] = evaluator.evaluate_standard()
        
    if args.eval_type in ['hard', 'all']:
        print("Running hard evaluation...")
        results['hard'] = evaluator.evaluate_hard()
        
    if args.eval_type in ['ood', 'all']:
        print("Running OOD evaluation...")
        results['ood'] = evaluator.evaluate_ood()
        
    if args.eval_type in ['adversarial', 'all']:
        print("Running adversarial evaluation...")
        results['adversarial'] = adv_evaluator.evaluate()
        
    # Save results
    model_name = os.path.basename(args.model_path).replace('.pth', '')
    res_path = os.path.join(out_dir, f"eval_{model_name}.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Evaluation complete. Results saved to {res_path}")
    print("Summary of results:")
    for k, v in results.items():
        print(f"  {k.capitalize()}: {v}")

if __name__ == "__main__":
    main()
