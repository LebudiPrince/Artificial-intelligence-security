import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1,self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

    def num_flat_features(self, x):
        size=x.size()[1:]
        n_features=1
        for i in size:
            n_features*=i
        return n_features


def train(batch_size, lr=0.001):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='F:/CIFAR', train=True, download=False, transform=transform)
    trainloader=torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    net=Net()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    for eppch in range(10):
        for i,data in enumerate(trainloader,0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs=net(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            print(eppch,"-",i+1,loss.item())

    print("Finishing Training! ")
    torch.save(net, "./model_adam.pkl")


def test(batch_size):
    net = torch.load('model_adam.pkl')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='F:/CIFAR', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    total = 0
    right = 0
    with torch.no_grad():
        for data in testloader:
            total += batch_size
            images, labels = data
            outputs = net(images)
            rate, index = torch.max(outputs.data, 1)
            print("data:")
            print(data)
            print("outputs")
            print(outputs)
            print("result:")
            print(rate,index)
            for i in range(batch_size):
                if (labels[i].item() == index[i]):
                    right += 1
    print("accuracy:", round(right / total, 2))

if __name__=='__main__':
    BATCHA_SIZE = 50
    train(BATCHA_SIZE)
    test(BATCHA_SIZE)