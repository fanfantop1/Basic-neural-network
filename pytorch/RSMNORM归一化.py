import torch
import torch.nn as nn




class RSMNorm(nn.Module):
    def __init__(self, dim, eps:float=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))



    def _norm(self,x):
        return torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+torch.tensor(self.eps))

    def forward(self, x):
        return self.weight*self._norm(x.float()).type_as(x)*x


 

RSMNorm = RSMNorm(dim=4)
input = torch.tensor([[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0]])
output = RSMNorm(input)
print(output)



