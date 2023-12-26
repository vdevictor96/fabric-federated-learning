import torch
import torch.nn as nn
import torch.optim as optim


class Perceptron(nn.Module):
    def __init__(self, n_features):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(n_features, 1)  # One output neuron

    def forward(self, x):
        return torch.sigmoid(self.fc(x))  # Using sigmoid activation function

# class Perceptron(nn.Module):
#     def __init__(self, n_features = 10, num_classes = 1):
#         super(Perceptron, self).__init__()
#         self.fc = nn.Linear(n_features, num_classes)


#     def forward(self, x):
#         out = torch.relu(self.fc(x))
#         return out

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size=32 * 32 * 3, hidden_layers=[50], num_classes=10):
        super(MultiLayerPerceptron, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(nn.ReLU())
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layers[-1], num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
