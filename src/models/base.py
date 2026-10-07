from abc import ABC, abstractmethod
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_feature_maps(self, x, layer_idx):
        pass

class ModelFactory:
    _registry = {}

    @classmethod
    def register(cls, name, model_class):
        cls._registry[name] = model_class

    @classmethod
    def create(cls, name, **kwargs):
        if name not in cls._registry:
            raise ValueError(f"Model {name} not registered")
        return cls._registry[name](**kwargs)

def get_model(name, **kwargs):
    return ModelFactory.create(name, **kwargs)
