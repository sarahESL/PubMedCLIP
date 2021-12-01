"""
Auto-encoder module for MEVF model
This code is written by Binh X. Nguyen and Binh D. Nguyen
<link paper>
"""
import torch.nn as nn
from torch.distributions.normal import Normal
import functools
import operator
import torch.nn.functional as F
import torch

def add_noise(images, mean=0, std=0.1):
    normal_dst = Normal(mean, std)
    noise = normal_dst.sample(images.shape)
    noisy_image = noise + images
    return noisy_image

def print_model(model):
    print(model)
    nParams = 0
    for w in model.parameters():
        nParams += functools.reduce(operator.mul, w.size(), 1)
    print(nParams)

class Auto_Encoder_Model(nn.Module):
    def __init__(self):
        super(Auto_Encoder_Model, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, 64, padding=1, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 32, padding=1, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 16, padding=1, kernel_size=3)

        # Decoder
        self.tran_conv1 = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.tran_conv2 = nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward_pass(self, x):
        output = F.relu(self.conv1(x))
        output = self.max_pool1(output)
        output = F.relu(self.conv2(output))
        output = self.max_pool2(output)
        output = F.relu(self.conv3(output))
        return output

    def reconstruct_pass(self, x):
        output = F.relu(self.tran_conv1(x))
        output = F.relu(self.conv4(output))
        output = F.relu(self.tran_conv2(output))
        output = torch.sigmoid(self.conv5(output))
        return output

    def forward(self, x):
        output = self.forward_pass(x)
        output = self.reconstruct_pass(output)
        return output
