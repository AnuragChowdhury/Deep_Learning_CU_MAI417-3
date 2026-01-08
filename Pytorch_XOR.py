import torch
import torch.nn as nn
import torch.optim as optim

# XOR Data
X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]], dtype=torch.float32)
y = torch.tensor([[0.],[1.],[1.],[0.]], dtype=torch.float32)

# Hyperparameters
learning_rate = 0.1
hidden_neurons = 4
epochs = 5000

# Layers (created directly, not inside a class)
W1 = nn.Linear(2, hidden_neurons)
W2 = nn.Linear(hidden_neurons, 1)

activation = nn.Tanh()

# Optimizer
optimizer = optim.SGD(list(W1.parameters()) + list(W2.parameters()), lr=learning_rate)

# Loss
criterion = nn.BCELoss()

# Training loop
for epoch in range(epochs):
    # Forward pass manually
    hidden = activation(W1(X))
    output = torch.sigmoid(W2(hidden))

    loss = criterion(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Test
hidden = activation(W1(X))
output = torch.sigmoid(W2(hidden))
print("Predictions:")
print(output.detach())
