import torch
import numpy as np
import torch.nn as nn
import time
from .flatten import Flatten

__all__ = ['B_AlexNet', 'b_alexnet']
# baseline_time = 0.002343075680732727
baseline_time = 0.0024395286321640015
avg_b_time = [0.0015824947834014893, 0.0022942611694335936, 0.003047058033943176]
# avg_b_time = [0.0014306637525558473, 0.002061791038513184, 0.0027997693777084354]


class B_AlexNet(nn.Module):

    def __init__(self, num_classes=10, avg_b_time=avg_b_time, baseline_time=baseline_time):
        super(B_AlexNet, self).__init__()
        self.avg_b_time = avg_b_time
        self.baseline_time = baseline_time
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
            nn.Linear(512, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, num_classes)
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
        b3 = self.classifier(x)
        return b1, b2, b3

class Alex_B1(nn.Module):

    def __init__(self, num_classes=10):
        super(Alex_B1, self).__init__()
        self.input_size = 32
        self.channel_num = 3
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
    
    def forward(self, x):
        x = self.feature1(x)
        x = self.branch1(x)
        return x

class Alex_B2(nn.Module):

    def __init__(self, num_classes=10):
        super(Alex_B2, self).__init__()
        self.input_size = 16
        self.channel_num = 64
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
    
    def forward(self, x):
        x = self.feature2(x)
        x = self.branch2(x)
        return x

class Alex_B3(nn.Module):

    def __init__(self, num_classes=10):
        super(Alex_B3, self).__init__()
        self.input_size = 8
        self.channel_num = 96
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
        x = self.feature3(x)
        x = self.classifier(x)
        return x


class AlexNet(nn.Module):

    def __init__(self, num_classes=10, avg_b_time=avg_b_time):
        super(AlexNet, self).__init__()
        self.input_size = 32
        self.channel_num = 3
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
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
        x = self.feature(x)
        x = self.classifier(x)
        return x 

def eval_model_time(model_name):
    model = model_name()
    model.eval()
    random_input = torch.randn(1, model.channel_num, model.input_size, model.input_size)
    time_list = []
    for _ in range(10001):
        tic = time.time()
        model(random_input)
        time_list.append(time.time() - tic)
    time_list = np.array(time_list[1:])
    return np.mean(time_list)


def eval_time_f():
    baseline_time = eval_model_time(AlexNet)
    models = [Alex_B1, Alex_B2, Alex_B3]
    avg_b_time = []
    pre_time = 0.0
    for i in models:
        pre_time += eval_model_time(i)
        avg_b_time.append(pre_time)
    return baseline_time, avg_b_time
        

def b_alexnet(pretrained=True, eval_time=False):
    global avg_b_time, baseline_time
    if eval_time:
        baseline_time, avg_b_time = eval_time_f()
    model = B_AlexNet(avg_b_time=avg_b_time, baseline_time=baseline_time)
    if pretrained:
        # state_dict = torch.load('./pretrained_models/Alex_main_branch_best.pth')
        state_dict = torch.load('./pretrained_models/Alex_full_branch.pth')
        model.load_state_dict(state_dict, strict=True)
    return model

# if __name__ == "__main__":
#     print(avg_b_time)
#     avg_b_time = eval_time()
#     print(avg_b_time)



# print('==> Preparing data..')
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

# model = B_AlexNet()

# for batch_idx, (inputs, targets) in enumerate(trainloader):
#     output_b1, output_b2, output = model(inputs)
#     print(output_b1.size())
#     print(output_b2.size())
#     print(output.size())
#     print(inputs.size())
#     break
