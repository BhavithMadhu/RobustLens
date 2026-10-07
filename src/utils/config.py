import yaml
from dataclasses import dataclass, field, replace
from typing import Dict, Any, Optional

@dataclass
class ModelConfig:
    name: str = 'resnet18'
    pretrained: bool = False
    num_classes: int = 10
    kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrainingConfig:
    batch_size: int = 128
    learning_rate: float = 1e-3
    epochs: int = 50
    weight_decay: float = 1e-4
    optimizer: str = 'adam'

@dataclass
class ExperimentConfig:
    experiment_name: str
    seed: int = 42
    output_dir: str = './outputs'
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
def merge_configs(base: ExperimentConfig, override: Dict[str, Any]) -> ExperimentConfig:
    base_dict = {
        'experiment_name': override.get('experiment_name', base.experiment_name),
        'seed': override.get('seed', base.seed),
        'output_dir': override.get('output_dir', base.output_dir),
    }
    
    if 'model' in override:
        model_overrides = override['model']
        model_kwargs = {k: v for k, v in model_overrides.items() if hasattr(ModelConfig, k)}
        model_kwargs['kwargs'] = model_overrides.get('kwargs', base.model.kwargs)
        base_dict['model'] = replace(base.model, **model_kwargs)
        
    if 'training' in override:
        train_overrides = override['training']
        train_kwargs = {k: v for k, v in train_overrides.items() if hasattr(TrainingConfig, k)}
        base_dict['training'] = replace(base.training, **train_kwargs)
        
    return replace(base, **base_dict)

def load_config(path: str) -> ExperimentConfig:
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    model_cfg = ModelConfig(**config_dict.get('model', {}))
    train_cfg = TrainingConfig(**config_dict.get('training', {}))
    
    return ExperimentConfig(
        experiment_name=config_dict.get('experiment_name', 'default_exp'),
        seed=config_dict.get('seed', 42),
        output_dir=config_dict.get('output_dir', './outputs'),
        model=model_cfg,
        training=train_cfg
    )
