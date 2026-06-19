import torch


n=100
D_in=1
D_out=1

x_data = torch.randn(n,D_in)
t_W = torch.tensor([[2.0]])
t_b = torch.tensor(2.0)

y_data = x_data@t_W + t_b + torch.randn(n,D_out)

w = torch.randn(D_out,D_in,requires_grad=True)
b = torch.randn(1,requires_grad=True)
print(w,b)
y_hat = x_data @ w + b
print(y_data)
print(y_hat)

error = y_data-y_hat
squared_error = error**2
loss = squared_error.mean()
print(loss)
loss.backward()
print(w.grad)
print(b.grad)

learning_rate , epochs = 0.01 , 9

w,b = torch.randn(1,1,requires_grad=True),torch.randn(1,requires_grad=True)

for epoch in range(epochs):

    y_hat = x_data @ w + b
    loss = torch.mean((y_data-y_hat)**2)
    loss.backward()
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -=  learning_rate * b.grad
        print(w,b)
        print("LOss", loss)
    w.grad.zero_()
    b.grad.zero_()


print("LOss",loss)







