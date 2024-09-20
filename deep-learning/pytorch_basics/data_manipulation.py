import torch

x = torch.arange(12, dtype=torch.float32)
x.shape

x.reshape(3, 4)
x.reshape(-1, 4)

# This is the same as the last dimension is inferred when using -1
x.reshape(2, 2, 3)
x.reshape(2, 2, -1)

x = torch.randn(3, 4)


x = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

# Last row
x[-1]

# First column
x[:, 0]

# Last row, first two columns
x[-1, :2]

x[:, -1]

torch.exp(x)

# Operations
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x**y

# Concatenation
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
concatenated_rows = torch.cat((X, Y), dim=0)
concatenated_columns = torch.cat((X, Y), dim=1)

# Broadcasting. Need to be careful here as it might not be what you expect
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
a + b

# What happens in more dimensions?
a = torch.arange(3).reshape((3, 1, 1))
b = torch.arange(2).reshape((1, 2, 1))
a, b
a + b

# Allocating to the same memory
Z = torch.zeros_like(Y)
print("id(Z):", id(Z))
Z[:] = X + Y
print("id(Z):", id(Z))

# Reuse X to save memory
before = id(X)
X += Y
id(X) == before

# Summing tensors
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
A.sum()
A.sum(axis=0)
a_sum_rows = A.sum(axis=0, keepdim=True)
a_sum_columns = A.sum(axis=1, keepdim=True)

# Dot product
x = torch.randn(3, dtype=torch.float32)
y = torch.ones(3, dtype=torch.float32)
x, y, torch.dot(x, y)

# Matrix-vector products
A.shape, x.shape, torch.mv(A, x), A @ x

# Matrix-matrix multiplication
A.shape
B = torch.ones(3, 4)
torch.mm(A, B), A @ B

# Plotting matrix multiplication as projections
import matplotlib.pyplot as plt

tensor = torch.rand(10, 10)
tensor_np = tensor.cpu().numpy()
plt.imshow(tensor_np, cmap="viridis")  # You can choose a different colormap
plt.colorbar()  # Adds a colorbar to the side for reference
plt.title("2D Tensor Visualization")
plt.show()

# Norms
torch.norm(torch.ones((4, 9)))

# TODO: Play around with the spectral norm.
# It seems that spectral norm checks how much it can stretch a vector.
# It is dependent on the vector since it is connected to the largest eigenvalue?
