import torch

print(torch.ones(10))
print(torch.ones(10).size())

#Arange
#arange(start, end, step)
a = torch.arange(0,5,1)
print(a)

b = torch.arange(0,5,2)
print(b)

#View
#Reshape without copying memory
x = torch.arange(12)           # shape (12,)
print("View x", x)
y = x.view(3, 4)               # shape (3, 4), shares storage with x
print("View x is reshaped to y", y)
y[0,0] = 999                   # -> tensor(999)  # proves shared storage
print("View print x but with change made to y", x)
print("View print y after change made to y", y)

#View vs Reshape
# view(new_shape): zero-copy if contiguous; errors otherwise.
# reshape(new_shape): tries zero-copy; if not possible, it silently copies (so it won’t error on non-contiguous input).

#Transpose
x = torch.randn(2, 3, 4)        # (B=2, T=3, D=4)
y = x.transpose(1, 2)           # shape -> (2, 4, 3)
print("Transpose original tensor", x.shape)
print("Transpose transposed tensor", y.shape)

#Matrix multiplication
p = torch.randn(5, 4, 3)
q = torch.randn(5, 3, 6)
y = p @ q
print("Batched matrix multiplication", y.shape)  # torch.Size([5, 4, 6])

#size
print(p.size(0))

#mask
print("tensor", b)
print("mask", (b != 0))
print("mask", (b != 0).unsqueeze(0))
print("mask", (b != 0).unsqueeze(0).int())

#causal mask
print(torch.triu(torch.ones(2,2)))
print(torch.triu(torch.ones(1,2,2)))
print((torch.triu(torch.ones(2,2)) == 0).int())