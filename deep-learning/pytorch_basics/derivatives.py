import torch

x = torch.arange(4.0)
x

2 * x @ x

# We will compute derivatives for each dimension of the vector
x.requires_grad_(True)
x.grad  # The gradient is None by default

y = 2 * torch.dot(x, x)
y

# Calling backward() computes the derivative of y with respect to each component of x
y.backward()
x.grad

# Reset gradient
x.grad.zero_()  # Reset the gradient
y = x.sum()
y.backward()
x.grad
