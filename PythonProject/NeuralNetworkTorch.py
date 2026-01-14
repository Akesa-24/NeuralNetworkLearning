import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

class Model(nn.Module):

# input layer, 4 features of the flower -> Hidden layer


    def __init__(self, in_features=4, h1=8, h2=9, h3=5, out_features=3, *args, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x

torch.manual_seed(37)
model = Model()

url = ''
my_df = pd.read_csv(url)