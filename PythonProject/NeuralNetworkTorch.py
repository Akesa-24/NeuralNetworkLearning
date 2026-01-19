import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
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

torch.manual_seed(32)
model = Model()

url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
my_df = pd.read_csv(url)

my_df['species'] = my_df['species'].replace({
    'setosa': 0,
    'versicolor': 1,
    'virginica': 2
})

# my_df['species'] = my_df['species'].replace('setosa', 0.0)
# my_df['species'] = my_df['species'].replace('versicolor', 1.0)
# my_df['species'] = my_df['species'].replace('virginica', 2.0)


X = my_df.drop('species', axis=1).values
y = my_df['species'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 32 )

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training the model
epochs = 100
losses = []
for i in range(epochs):
    # Forward pass
    y_pred = model.forward(X_train)

    # Compute Loss
    loss = criterion(y_pred, y_train)
    losses.append(loss.detach().numpy())

    #print every 10 epochs
    if i % 10 == 0:
        print(f'Epoch {i} Loss: {loss}')

    # Back propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.show()

with torch.no_grad():
    y_eval = model.forward(X_test)
    loss = criterion(y_eval, y_test)
    print(f'Test Loss: {loss}')
    y_pred = torch.argmax(y_eval, axis=1)
    acc = (y_pred == y_test).sum().item() / len(y_test)
    print(f'Accuracy: {acc*100:.2f}%')
