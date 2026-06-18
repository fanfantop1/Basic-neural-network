
import torch


D_in=1
D_out=1

linear = torch.nn.Linear(D_in,D_out)

print(linear.weight)
print(linear.bias)

#y_hat = linear(X)


#常用的激活函数
relu = torch.nn.ReLU()
data = torch.randn(3,3)
data_1 = relu(data)
print(data)
print(data_1)

gelu = torch.nn.GELU()
data_2 = gelu(data)
print(data)
print(data_2)

softmax = torch.nn.Softmax(dim=0)

data_3 = softmax(torch.randn(2,2,2,2))

print(data_3)

#词嵌入
vocab_size = 6
vocab_dim = 3


vocab_layer = torch.nn.Embedding(vocab_size,vocab_dim)

word = torch.tensor([[1,2,3,4,5],[2,2,4,4,4]])

word_embedding = vocab_layer(word)
print(word_embedding)




