import torch


shape = (4,9)
shape2 = (9,4)


tensor_a = torch.randn(shape,requires_grad = True)
tensor_b = torch.randn(shape2,requires_grad = True)


tensor_c = tensor_a @ tensor_b


print(tensor_c)
print(tensor_a)
print(tensor_b)


mean_tensora = tensor_a.sum(dim=0)
print(mean_tensora)



