import torch
import numpy as np
import torch.nn as nn
import torchvision
import urllib
from PIL import Image
from torchvision import transforms
from collections import OrderedDict
from flatten import Flatten
# from torch.hub import load_state_dict_from_url
# print(os.path.dirname(__file__))

class B_AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(B_AlexNet, self).__init__()
        self.feature1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        )
        self.branch1 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Flatten(),
            nn.Linear(512, num_classes),
        )
        self.feature2 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1)
        )
        self.branch2 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(96, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Flatten(),
            nn.Linear(128, num_classes),
        )
        self.feature3 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.feature1(x)
        b1 = self.branch1(x)
        x = self.feature2(x)
        b2 = self.branch2(x)
        x = self.feature3(x)
        x = self.classifier(x)
        return b1, b2, x

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

model = B_AlexNet()

for batch_idx, (inputs, targets) in enumerate(trainloader):
    output_b1, output_b2, output = model(inputs)
    print(output_b1.size())
    print(output_b2.size())
    print(output.size())
    print(inputs.size())
    break
