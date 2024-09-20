import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LinearRegression

torch.manual_seed(0)

# Data parameters. Since this is a simple linear model the optimization should
# be able to find the optimal values of the weights and bias. Higher number of samples
# will make the optimization more stable.

# Under certain conditions, the SGD optimization finds the global optimum for a convex
# function.
w = torch.tensor([2, -3.4])
b = 4.2
noise = 0.01
num_train = 100
num_val = 100

n = num_train + num_val
X = torch.randn(n, len(w))
noise = torch.randn(n, 1) * noise
y = torch.matmul(X, w.reshape((-1, 1))) + b + noise

# Initialize estimates randomly.
w_est = torch.randn(len(w), 1, requires_grad=True)
b_est = torch.randn(1, requires_grad=True)

# Batch SGD hyperparameters.
lr = 0.03
num_epochs = 100
batch_size = 32


# Prepare data splits.
dataset = TensorDataset(X, y)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# Shuffle=True shuffles at each epoch. I assume that the data is shuffled each time the
# for _ in data_loader is called.

for epoch in range(num_epochs):
    for X_batch, y_batch in data_loader:
        batch_loss = (y_batch - torch.mm(X_batch, w_est) - b_est).pow(2).sum()
        batch_loss.backward()  # Compute gradients for all parameters that have requires_grad=True.

        with torch.no_grad():  # Disable gradient tracking.
            w_est -= (lr / batch_size) * w_est.grad
            b_est -= (lr / batch_size) * b_est.grad

        w_est.grad.zero_()
        b_est.grad.zero_()

        print(w_est)
        print(b_est)

# Compare the estimated values with the true values.
reg = LinearRegression().fit(X.numpy(), y.numpy())
print(reg.coef_)
print(reg.intercept_)
print(b_est)
print(w_est)
