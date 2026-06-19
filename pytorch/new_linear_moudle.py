from use_torch_nn import Linear_Moudle
import torch




D_out = 1

D_in=1

linear = Linear_Moudle(1,1)

epochs = 10000

n=100

w= torch.tensor([[2.0]])

b = torch.tensor(2)

x_data = torch.randn(n,D_in)


y =  x_data @ w + b + torch.randn(n,D_out)


learning_rate = 0.01
optimizer = torch.optim.Adam(linear.parameters(), lr=learning_rate)
loss_func = torch.nn.MSELoss()


for epoch in range(epochs):
    y_pred = linear(x_data)
    loss = loss_func(y_pred,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss)