import torch
import torch.nn as nn

class CheaterCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(CheaterCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=9, stride=2, padding=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=2, padding=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_feature_maps(self, x, layer_idx):
        features = []
        x = self.conv1(x)
        x = self.relu1(x)
        features.append(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        features.append(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        features.append(x)
        
        if layer_idx < len(features):
            return features[layer_idx]
        else:
            raise ValueError(f"Invalid layer index: {layer_idx}")
