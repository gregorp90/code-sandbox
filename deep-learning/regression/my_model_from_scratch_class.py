from typing import Optional

import torch
import torch.nn as nn


class LinearRegressionScratch(nn.Module):
    """The linear regression model implemented from scratch."""

    def __init__(self, num_inputs: int, sigma: float = 0.01):
        super().__init__()
        self.num_inputs = num_inputs
        self.sigma = sigma
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, X: torch.tensor) -> torch.tensor:
        return torch.mm(X, self.w) + self.b

    def loss(self, y_hat: torch.tensor, y: torch.tensor) -> torch.tensor:
        return (y - y_hat).pow(2).mean()


class SGDOptimizer:
    def __init__(self, params: list[torch.tensor], lr: float):
        self.params = params
        self.lr = lr

    def step(self):
        with torch.no_grad():
            for param in self.params:
                param -= self.lr * param.grad

    def zero_grad(self):
        with torch.no_grad():
            for param in self.params:
                param.grad.zero_()


class SGDOptimizerFactory:
    def __init__(self, lr: float):
        self.lr = lr

    def with_params(self, params: list[torch.tensor]) -> SGDOptimizer:
        return SGDOptimizer(params, self.lr)


# So we have an instance of a model, and then we can train it with different trainers.
# The model holds the current state of the weights and biases.
class Trainer:
    def __init__(
        self,
        optimizer_factory: SGDOptimizerFactory = SGDOptimizerFactory(lr=0.03),
        num_epochs: int = 100,
    ):
        self.optimizer_factory = optimizer_factory
        self.num_epochs = num_epochs

    def fit(
        self,
        model: LinearRegressionScratch,
        data_loader: torch.utils.data.DataLoader,
        data_loader_val: Optional[torch.utils.data.DataLoader],
    ) -> dict:
        optimizer = self.optimizer_factory.with_params([model.w, model.b])
        epoch_num = []
        loss_train = []
        loss_val = []

        for epoch in range(self.num_epochs):
            epoch_num.append(epoch)

            model.train()  # Prepares the model for training
            epoch_loss = 0
            for X_batch, y_batch in data_loader:
                y_hat = model(X_batch)  # Short for model.forward(X_batch)
                loss = model.loss(y_hat, y_batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
                print([model.w, model.b])

            print("Training loss:", epoch_loss)
            loss_train.append(epoch_loss)

            if data_loader_val is not None:
                with torch.no_grad():
                    model.eval()
                    epoch_val_loss = 0

                    for X_batch, y_batch in data_loader_val:
                        y_hat = model(X_batch)
                        loss = model.loss(y_hat, y_batch)
                        epoch_val_loss += loss.item()

                    print("Validation loss:", epoch_val_loss)
                    loss_val.append(epoch_val_loss)

        return {"epoch_num": epoch_num, "loss": loss_train, "loss_val": loss_val}
