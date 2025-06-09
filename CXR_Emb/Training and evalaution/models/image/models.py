import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models


class DensNetWithHead(nn.Module):
    def __init__(self,  hidden_layer_sizes, dropout_rate, num_classes):
        super(DensNetWithHead, self).__init__()

        # Pretrained DenseNet backbone
        self.backbone = models.densenet121(pretrained=True)
        num_features = self.backbone.classifier.in_features

        # Remove the last classification layer of the backbone
        self.backbone.classifier = nn.Identity()

        # Custom head with hidden layers
        layers = []
        input_size = num_features

        for size in hidden_layer_sizes:
            linear_layer = nn.Linear(input_size, size)
            init.kaiming_uniform_(linear_layer.weight, nonlinearity='relu')
            layers.append(linear_layer)
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.Dropout(dropout_rate))
            input_size = size

        # Output layer
        layers.append(nn.Linear(input_size, num_classes))

        # Assemble the custom head
        self.custom_head = nn.Sequential(*layers)

    def forward(self, x):
        # Forward pass through the backbone
        features = self.backbone(x)

        # Forward pass through the custom head
        output = self.custom_head(features)

        return output
