import torch.nn as nn
from .base import BaseModel

class RobustCNN(BaseModel):
    def __init__(self, num_classes: int = 10, batch_norm: bool = True, use_batchnorm: bool = None, dropout_rate: float = 0.0):
        super(RobustCNN, self).__init__()
        if use_batchnorm is not None:
            batch_norm = use_batchnorm
        self.batch_norm = batch_norm
        
        # Conv block 1
        self.conv1 = nn.Conv2d(3, 8, kernel_size=9, stride=2, padding=4)
        self.bn1 = nn.BatchNorm2d(8) if batch_norm else nn.Identity()
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Conv block 2
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(16) if batch_norm else nn.Identity()
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Conv block 3
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(16) if batch_norm else nn.Identity()
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        self.fc = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.drop3(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_feature_maps(self, x, layer_idx):
        features = []
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        features.append(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        features.append(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.drop3(x)
        features.append(x)
        
        if layer_idx < len(features):
            return features[layer_idx]
        else:
            raise ValueError(f"Invalid layer index: {layer_idx}")
