import torch



shape = (4,9)

tensor = torch.rand(shape,requires_grad = True)

tensor_x = tensor[:,2]
tensor_best = torch.argmax(tensor,dim=0)
tensor_y = torch.tensor([[1],[2],[3],[3],[3]])
tensor_c = torch.argmax(tensor,dim=0)
tensor_z = torch.gather(tensor,index = tensor_y,dim=0)

print(tensor_x)