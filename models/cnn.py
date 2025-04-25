import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=12, dropout_rate = 0.5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Input: (1, 128, 128)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # Output: (32, 64, 64)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: (64, 64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # Output: (64, 32, 32)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Output: (128, 32, 32)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # Output: (128, 16, 16)

        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # (32, 64, 64)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # (64, 32, 32)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))  # (128, 16, 16)
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
