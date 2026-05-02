import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from mnist_model import Network,nn
import torch.optim as optim


print("PyTorch 版本:", torch.__version__)
print("CUDA 是否可用:", torch.cuda.is_available())
print("当前使用的 CUDA 版本:", torch.version.cuda)
print("GPU 设备名称:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无 GPU")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用{device}计算")
if __name__ == '__main__':
    """数据读取"""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(root='C:/Users/fanfan\Desktop\手搓神经网络\mnist\mnist_images/train',  transform=transform)
    test_data = datasets.ImageFolder(root='C:/Users/fanfan\Desktop\手搓神经网络\mnist\mnist_images/test', transform=transform)
    print("train_data",len(train_data))
    print("test_data", len(test_data))
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    print("train_loader", len(train_loader))
    for batch_idx, (data, label) in enumerate(train_loader):
        if batch_idx ==3:
            break
        print("batch_idx", batch_idx)
        print("data_shape", data.shape)
        print("label", label)

    print("完数据读取")


    model = Network().to(device)#神经网络
    optimizer = optim.Adam(model.parameters())#优化器
    criterion = nn.CrossEntropyLoss()#损失函数
    for epoch in range(10):
        for batch_idx, (data, label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = criterion(output, label)
            optimizer.step()
            optimizer.zero_grad()

            if batch_idx % 100 == 0:
                print(f"| 训练轮数:{epoch + 1}/{10}",
                      f"| batch:{batch_idx}/{len(train_loader)}",
                      f"| Loss:{loss.item()} |")

    torch.save(model.state_dict(), "mnist.pth")
    print("训练完成")