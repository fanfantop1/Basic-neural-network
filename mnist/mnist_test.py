from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Grayscale

from mnist_model import Network
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                    transforms.ToTensor()
                                    ])
    test_data = datasets.ImageFolder(root='C:/Users/fanfan\Desktop\手搓神经网络\mnist\mnist_images/test',  transform=transform)
    print("test_set",len(test_data))
    model = Network()
    model.load_state_dict(torch.load('mnist.pth'))

    right = 0
    for i,(data,label) in enumerate(test_data):

        output = model(data)
        prediction = torch.argmax(output,1).item()
        if prediction == label:
            right += 1
        else:
            img_path = test_data.samples[i][0]
            print("正确数字",label,"预测数字",prediction,"图像路径",img_path)

acc =right/len(test_data)
print(f"准确率{right}/{len(test_data)},{acc}")
