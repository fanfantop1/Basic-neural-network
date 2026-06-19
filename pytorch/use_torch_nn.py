
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


#nn层归一化
norm_layer = torch.nn.LayerNorm(normalized_shape=vocab_dim)
norm_layer_features = norm_layer(word_embedding)
print(norm_layer_features)

#nn.dropout
dorpout = torch.nn.Dropout(0.6)
input = torch.ones(5,5)


dorpout.train()
dropout_train = dorpout(input)
print(dropout_train)


#nn.Moudle


class Linear_Moudle(torch.nn.Module):
    def __init__(self, in_put, out_put):
        super().__init__()
        self.Linear = torch.nn.Linear(in_put, out_put)

    def forward(self, x):
        return self.Linear(x)




print(linear)


#nn.optim

learning_rate = 0.01
optimizer = torch.optim.Adam(linear.parameters(), lr=learning_rate)
loss_func = torch.nn.MSELoss()











