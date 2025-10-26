import torch
import pandas
import numpy
import matplotlib as plt    
import torch.nn as nn
import torch.nn.functional as F

class ChessNet(nn.Module):


    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))




