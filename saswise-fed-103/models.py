import torch
from collections import OrderedDict

# Assuming ResNet20, evaluate_model, etc., are in utils2.py
# If not, their definitions need to be added here or in utils2.py
try:
    from utils2 import ResNet20, evaluate_model, compute_confusion_matrix, plot_confusion_matrix
except ImportError:
    # Provide dummy implementations or raise a clearer error
    print("Warning: utils2.py not found or incomplete. Using dummy model functions.")
    import torch.nn as nn
    # Define a very simple dummy model if ResNet20 is not found
    class ResNet20(nn.Module):
        def __init__(self, num_classes=10):
            super(ResNet20, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self._make_layer(16, 16, 3)
            self.layer2 = self._make_layer(16, 32, 3, stride=2)
            self.layer3 = self._make_layer(32, 64, 3, stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64, num_classes)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        def _make_layer(self, in_planes, planes, blocks, stride=1):
            layers = []
            layers.append(BasicBlock(in_planes, planes, stride))
            for _ in range(1, blocks):
                layers.append(BasicBlock(planes, planes))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

    class BasicBlock(nn.Module):
        def __init__(self, in_planes, planes, stride=1):
            super(BasicBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes)
                )

        def forward(self, x):
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = self.relu(out)
            return out

    # Dummy evaluation functions
    def evaluate_model(net, testset, device="cpu"):
        print("Warning: Using dummy evaluate_model function.")
        return 0.5, 0.75 # Dummy loss and accuracy

    def compute_confusion_matrix(net, testset, device="cpu"):
        print("Warning: Using dummy compute_confusion_matrix function.")
        return [[0]*10 for _ in range(10)] # Dummy matrix

    def plot_confusion_matrix(cm, title):
        print(f"Warning: Skipping plot_confusion_matrix for title: {title}")


def get_weights(net):
    """Retrieves the parameters (weights) from the model."""
    ndarrays = [
        val.cpu().numpy() for _, val in net.state_dict().items()
    ]
    return ndarrays

def set_weights(net, parameters, device):
    """Sets the parameters (weights) of the model."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {k: torch.tensor(v).to(device) for k, v in params_dict}
    )
    net.load_state_dict(state_dict, strict=True) 