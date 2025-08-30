import torch
import torch.nn as nn

class DrivingModel(nn.Module):
    def __init__(self):
        super(DrivingModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 64),  # 10 input features (ví dụ vị trí xe NPC)
            nn.ReLU(),
            nn.Linear(64, 3)   # 3 output: left / right / none
        )

    def forward(self, x):
        return self.fc(x)