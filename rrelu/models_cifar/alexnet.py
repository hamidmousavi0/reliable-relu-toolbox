import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
from rrelu.relu_bound.bound_relu import Relu_bound
class AlexNet_model(nn.Module):
    def __init__(self, n_classes=10,dropout_rate=0.0):
        super(AlexNet_model, self).__init__()
        self.conv1 =  nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 =  nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        self.maxpool3 =  nn.MaxPool2d(kernel_size=2)
        self.linear1 = nn.Linear(256 * 8 * 8, 4096)
        self.relu6 = nn.ReLU()
        self.linear2 = nn.Linear(4096, 4096)
        self.relu7= nn.ReLU()
        self.linear3 = nn.Linear(4096, n_classes)
        self.relu8= nn.ReLU()
        self.adaptpool = nn.AdaptiveAvgPool2d((8, 8))
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu2(self.conv2(x))
        x = self.maxpool2(x)
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.maxpool3(x)
        x = self.adaptpool(x) 
        x = x.view(x.size(0), -1)
        x = self.relu6(self.linear1(x))
        x = self.relu7(self.linear2(x))
        x = self.linear3(x)
        x = self.relu8(x)
        return x        

        
###########################################

