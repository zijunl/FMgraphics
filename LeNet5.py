
## Construct LeNet5 CNN structure

import torch

class LeNet5(torch.nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 5)
        self.conv2 = torch.nn.Conv2d(32, 64, 5)
        self.fc1 = torch.nn.Linear(64*5*5,200)
        self.fc2 = torch.nn.Linear(200, 100)
        self.fc3 = torch.nn.Linear(100, 2)
        self.activation = torch.nn.ReLU()
        self.activation2 = torch.nn.Sigmoid()
    def forward(self, x_in):
        x_in = self.activation(self.conv1(x_in))
        x_in = torch.nn.functional.max_pool2d(x_in,(2,2))
        x_in = self.activation(self.conv2(x_in))
        x_in = torch.nn.functional.max_pool2d(x_in,(2,2))
        x_in = x_in.view(-1, 64*5*5)
        x_in = self.activation(self.fc1(x_in))
        x_in = self.activation(self.fc2(x_in))
        x_out = self.fc3(x_in)
        return x_out

nets = LeNet5()
pytorch_total_params = sum(p.numel() for p in nets.parameters() if p.requires_grad)
#print(nets)
#print(pytorch_total_params)