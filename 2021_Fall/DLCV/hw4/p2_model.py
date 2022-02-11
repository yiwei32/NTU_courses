import torch.nn as nn

hidden_dims = 1000
num_classes = 65

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(hidden_dims, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.layer(x)
        