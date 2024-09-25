import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LinearRegression

from regression.my_model_from_scratch_class import LinearRegressionScratch, Trainer, LinearRegressionConcise, \
    SGDOptimizerFactory

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
noise_tensor = torch.randn(n, 1) * noise
y = torch.matmul(X, w.reshape((-1, 1))) + b + noise_tensor

X_val = torch.randn(n, len(w))
noise_val = torch.randn(n, 1) * noise
y_val = torch.matmul(X_val, w.reshape((-1, 1))) + b + noise_val

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
data_loader_val = DataLoader(
    TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=True
)

lr_model = LinearRegressionScratch(num_inputs=2, sigma=1)
trainer = Trainer(num_epochs=100)
loss_data = trainer.fit(lr_model, data_loader, data_loader_val)
print(loss_data)

print(lr_model.w)
print(lr_model.b)

reg = LinearRegression().fit(X.numpy(), y.numpy())
print(reg.coef_)
print(reg.intercept_)
print(lr_model.w)
print(lr_model.b)

########################################################################################
# This should also work. So we train the model with a 1-epoch trainer but 100 times.
# Which should be equivalent to the previous code.
########################################################################################
lr_model2 = LinearRegressionScratch(num_inputs=2, sigma=1)

for i in range(100):
    trainer = Trainer(num_epochs=1)
    trainer.fit(lr_model2, data_loader, data_loader_val)

print(lr_model2.w)  # Ok
print(lr_model2.b)  # Ok


########################################################################################
# Concise.
########################################################################################
lr_model = LinearRegressionConcise()
trainer = Trainer(num_epochs=100, optimizer_factory=SGDOptimizerFactory(lr=0.03, type=1))
# trainer = Trainer(num_epochs=100)  # Also works.

loss_data = trainer.fit(lr_model, data_loader, data_loader_val)
print(loss_data)
