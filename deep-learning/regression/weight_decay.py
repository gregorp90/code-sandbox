import torch
from torch.utils.data import TensorDataset, DataLoader

from regression.my_model_from_scratch_class import LinearRegressionScratch, Trainer, LinearRegressionConcise, \
    SGDOptimizerFactory, AbstractModel
from visualization.train_vs_val_loss import plot_loss_vs_val_loss

torch.manual_seed(0)

# Data parameters. Since this is a simple linear model the optimization should
# be able to find the optimal values of the weights and bias. Higher number of samples
# will make the optimization more stable.

# Under certain conditions, the SGD optimization finds the global optimum for a convex
# function.
noise = 0.01
num_train = 20
num_val = 100
num_inputs = 200

w = torch.ones(num_inputs) * 0.01
# w = torch.ones(num_inputs) * 0.1
b = torch.ones(1) * 0.05
# n = num_train + num_val
X = torch.randn(num_train, len(w))
noise_tensor = torch.randn(num_train, 1) * noise

y = torch.matmul(X, w.reshape((-1, 1))) + b + noise_tensor

X_val = torch.randn(num_val, len(w))
noise_val = torch.randn(num_val, 1) * noise
y_val = torch.matmul(X_val, w.reshape((-1, 1))) + b + noise_val

# Batch SGD hyperparameters.
num_epochs = 10
batch_size = 5

# Prepare data splits.
dataset = TensorDataset(X, y)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
data_loader_val = DataLoader(
    TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=True
)


def l2_penalty(w: torch.tensor) -> torch.tensor:
    return torch.sum(w.pow(2)) / 2


class WeightDecayScratch(LinearRegressionScratch):
    """The linear regression model implemented from scratch."""

    def __init__(self, num_inputs: int, lambd: float, sigma: float):
        super().__init__(num_inputs, sigma)
        self.lambd = lambd

    def forward(self, X: torch.tensor) -> torch.tensor:
        return torch.mm(X, self.w) + self.b

    def loss(self, y_hat: torch.tensor, y: torch.tensor) -> torch.tensor:
        return (y - y_hat).pow(2).mean() + self.lambd * l2_penalty(self.w)  # We also take the derivative of the penalty.

    def get_params(self) -> list[torch.tensor]:
        return [self.w, self.b]


# How do we set learning rate? Below a LR of 0.03 is completely off, we get infinite
# loss. But with 0.01 it works.
model = WeightDecayScratch(num_inputs=num_inputs, lambd=0, sigma=0.01)
trainer = Trainer(num_epochs=num_epochs, optimizer_factory=SGDOptimizerFactory(lr=0.01))
loss_data = trainer.fit(model, data_loader, data_loader_val)
plot_loss_vs_val_loss(loss_data, log_scale=True)
print(l2_penalty(model.w))


model = WeightDecayScratch(num_inputs=num_inputs, lambd=3, sigma=0.01)
trainer = Trainer(num_epochs=num_epochs, optimizer_factory=SGDOptimizerFactory(lr=0.01))
loss_data = trainer.fit(model, data_loader, data_loader_val)
plot_loss_vs_val_loss(loss_data, log_scale=True)
print(l2_penalty(model.w))

# I think this is a bad example as there is actually nothing to be learned from this model?
# The variance of the weights is the same as the variance of the noise. So the model is not able to learn anything.
# I might be missing something.
# Actually the variance of the sum us more than the noise, so there might be something to learn.
# However, I'm not sure about this.
