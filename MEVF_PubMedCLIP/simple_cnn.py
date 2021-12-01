"""
MAML module for MEVF model
This code is written by Binh X. Nguyen and Binh D. Nguyen
<link paper>
"""
import torch
import torch.nn as nn
import numpy as np
import pickle
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, weight_path='simple_cnn.weights', eps_cnn=1e-5, momentum_cnn=0.05):
        super(SimpleCNN, self).__init__()
        # init and load pre-trained model
        weights = self.load_weight(weight_path)
        self.conv1 = self.init_conv(1, 64, weights['conv1'], weights['b1'])
        self.conv1_bn = nn.BatchNorm2d(num_features=64, eps=eps_cnn, affine=True, momentum=momentum_cnn)
        self.conv2 = self.init_conv(64, 64, weights['conv2'], weights['b2'])
        self.conv2_bn = nn.BatchNorm2d(num_features=64, eps=eps_cnn, affine=True, momentum=momentum_cnn)
        self.conv3 = self.init_conv(64, 64, weights['conv3'], weights['b3'])
        self.conv3_bn = nn.BatchNorm2d(num_features=64, eps=eps_cnn, affine=True, momentum=momentum_cnn)
        self.conv4 = self.init_conv(64, 64, weights['conv4'], weights['b4'])
        self.conv4_bn = nn.BatchNorm2d(num_features=64, eps=eps_cnn, affine=True, momentum=momentum_cnn)

    def load_weight(self, path):
        return pickle.load(open(path, 'rb'))

    def forward(self, X):
        out = F.relu(self.conv1(X))
        out = self.conv1_bn(out)
        out = F.relu(self.conv2(out))
        out = self.conv2_bn(out)
        out = F.relu(self.conv3(out))
        out = self.conv3_bn(out)
        out = F.relu(self.conv4(out))
        out = self.conv4_bn(out)
        out = out.view(-1, 64, 36)

        return torch.mean(out, 2)

    def convert_to_torch_weight(self, weight):
        return np.transpose(weight, [3, 2, 0, 1])

    def init_conv(self, inp, out, weight, bias, convert=True):
        conv = nn.Conv2d(inp, out, 3, 2, 1, bias=True)
        if convert:
            weight = self.convert_to_torch_weight(weight)
        conv.weight.data = torch.Tensor(weight).float()
        conv.bias.data = torch.Tensor(bias).float()
        return conv

if __name__ == '__main__':
    simple_cnn = SimpleCNN(weight_path='simple_cnn.weights', eps_cnn=1e-5, momentum_cnn=0.05)
    npo = np.random.random((3, 1, 84, 84))
    x = torch.tensor(npo, dtype=torch.float32).float()
    simple_cnn(x)
