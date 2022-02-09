import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_chn, out_channels=10, kernel_size=5, stride=1, padding=0, bias=True)
        self.pool1 = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU(True)

        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0, bias=True)
        self.drop2 = nn.Dropout(0.5)
        self.pool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU(True)

        self.fc1 = nn.Linear(in_features=320, out_features=50, bias=True)
        self.relu_fc1 = nn.ReLU(True)
        self.drop_fc1 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(in_features=50, out_features=out_chn, bias=True)

    def forward(self, x):
        # perform the usual forward pass
        x = self.relu1(self.pool1(self.conv1(x)))

        x = self.relu2(self.pool2(self.drop2(self.conv2(x))))

        x = x.view(-1, 320)

        x = self.drop_fc1(self.relu_fc1(self.fc1(x)))
        x = self.fc2(x)

        #x = torch.log_softmax(x, dim=1)

        return x