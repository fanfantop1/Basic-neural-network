import torch

#1
data = [[1,2,3],[4,5,6]]
my_tensor = torch.tensor(data)
print(my_tensor)


#2
shape = (3,3)
ones = torch.ones(shape)
zeros = torch.zeros(shape)
randon = torch.randn(shape,requires_grad=True)

print(randon)
print(ones)
print(zeros)
print(randon.requires_grad)



#3
rand_like = torch.ones_like(ones,dtype = torch.float,device = 'cpu',requires_grad = True)
print(rand_like)
print(rand_like.requires_grad)
print(rand_like)
print(rand_like.shape)
print(rand_like.dtype)
print(rand_like.device)


#test

a = torch.tensor(1.0,requires_grad = True)
b = torch.tensor(2.0,requires_grad = True)
c = torch.tensor(3.0,requires_grad = True)

d = a+b
e = c*d
print(e)