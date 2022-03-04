import torch

import numpy as np
import torch.nn as nn
from torch import cuda
import torch.nn.functional as F


class MyBatchNorm2d(nn.BatchNorm2d):
    """
    Re-implementing the BatchNorm2d to exclude the paddings from the calculation
    Originated from [https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py]
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True, padding_value=-1):
        super(MyBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.padding_value = padding_value
        
    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        mask = (input!=self.padding_value).float()
        # calculate running estimates
        if self.training:        
            input = input * mask # Make the padding 0 if it's not already
            mean = input.sum(dim=(0, 2, 3)) / mask.sum(dim=(0, 2, 3))
            m = mean[None, :, None, None]
            var = torch.sum(((input - m)*mask) ** 2, dim=(0, 2, 3)) / mask.sum(dim=(0, 2, 3))

            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input
    

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding_value=-1):
        super(ResBlock, self).__init__()
        self.normal = nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, 3, stride=stride, bias=False, padding=1),
            # MyBatchNorm2d(out_channels, padding_value=padding_value),
            # nn.LeakyReLU(),
            # nn.Conv2d(out_channels, out_channels, 3, stride=1, bias=False, padding=1),
            # MyBatchNorm2d(out_channels, padding_value=padding_value),
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, bias=False, padding=1),
            # nn.Dropout(.5),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),

            nn.Conv2d(out_channels, out_channels, 3, stride=1, bias=False, padding=2, dilation=2),
            # nn.Dropout(.5),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),

            nn.Conv2d(out_channels, out_channels, 3, stride=1, bias=False, padding=4, dilation=4),
            # nn.Dropout(.5),
            nn.BatchNorm2d(out_channels),
        )



    def forward(self, x, mode=0):
        x1 = self.normal(x)
        if mode == 0:
            return F.leaky_relu(x1 + x)
        else: # mode == 1 or 2 (softmax on -1 or -2 dim)
            return F.softmax(x1 + x, dim=-mode)


class ResNet(torch.nn.Module):
    def __init__(self, padding_value=-1):
        super(ResNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(8, 32, 3, padding=1, bias=False),
            # nn.Dropout(.5),
            MyBatchNorm2d(32, padding_value=padding_value),
            nn.LeakyReLU(),
            
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            # nn.Dropout(.5),
            MyBatchNorm2d(32, padding_value=padding_value),
            nn.LeakyReLU(),

            ResBlock(32, 32, padding_value=padding_value),
            ResBlock(32, 32, padding_value=padding_value),
            # ResBlock(32, 32, padding_value=padding_value),
            # ResBlock(32, 32, padding_value=padding_value),
            # ResBlock(32, 32, padding_value=padding_value),
            # ResBlock(32, 32, padding_value=padding_value),
            # ResBlock(32, 32, padding_value=padding_value),
            # ResBlock(32, 32, padding_value=padding_value),
            # ResBlock(32, 32, padding_value=padding_value),
            # ResBlock(32, 32, padding_value=padding_value)
            )
        
        self.conv_shared = ResBlock(32, 32, padding_value=padding_value)
        
        self.readout = nn.Sequential(
            nn.Conv2d(32, 2, 1, bias=False),
            MyBatchNorm2d(2, padding_value=padding_value),
            nn.LeakyReLU(),

            nn.Conv2d(2, 1, 1),
        )
    
    def forward(self, x, test=False, repeat=None):
        n = x.size(-1)
        if repeat is None:
            repeat = 8 - min((x.shape[-1])//100, 4)
        outputs = []
        
        mask = x[:, -1].clone().unsqueeze(1)
        mask[mask==1] = -1000
        shared_x = self.conv(x)

        # mode = [0, 0, 1, 0, 0, 2]  # Alternating between softmax and relu
        # mode = [1, 2, 1, 2, 1, 2] # Only softmax
        mode = [0, 0, 0, 0, 0, 0] # Always relu
        for i in range(repeat):
            shared_x = self.conv_shared(shared_x, mode=mode[i%6])
            
            out = self.readout(shared_x)
            out = out + mask
            out = 0.5*(out + out.transpose(-1, -2))
            # out = F.sigmoid(out)
            out = F.softmax(out, dim=-1)

            # We need softmax at the end due to having binary matrix at the end
            # sum of each row == 1
            
            outputs.append(out)
        if test:
            return out.view(-1, 1, n, n)
        return torch.cat(outputs, dim=0)


class ResNetUnrolled(torch.nn.Module):
    def __init__(self, padding_value=-1):
        super(ResNetUnrolled, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(8, 32, 3, padding=1, bias=False),
            MyBatchNorm2d(32, padding_value=padding_value),
            nn.LeakyReLU(),
            
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            MyBatchNorm2d(32, padding_value=padding_value),
            nn.LeakyReLU(),
            )
        
        self.conv_shared = nn.ModuleList(ResBlock(32, 32, padding_value=padding_value) for i in range(4)) 
        
        self.readout = nn.Sequential(
            nn.Conv2d(32, 2, 1, bias=False),
            MyBatchNorm2d(2, padding_value=padding_value),
            nn.LeakyReLU(),

            nn.Conv2d(2, 1, 1),
        )
    
    def forward(self, x, test=False, repeat=None):
        n = x.size(-1)
        if repeat is None:
            repeat = 8 - min((x.shape[-1])//100, 4)
        outputs = []
        
        mask = x[:, -1].clone().unsqueeze(1)
        mask[mask==1] = -1000
        shared_x = self.conv(x)

        # mode = [0, 0, 1, 0, 0, 2] 
        # mode = [0, 0, 0, 0, 0, 0] # Always relu

        for i, layer in enumerate(self.conv_shared):
            shared_x = layer(shared_x, mode=0) # always with relu
        
        out = self.readout(shared_x)
        out = out + mask
        out = 0.5*(out + out.transpose(-1, -2))
        out1 = F.softmax(out, dim=-1)
        out2 = F.softmax(out, dim=-2)
        out = out1+out2

        # We need softmax at the end due to having binary matrix at the end
        # sum of each row == 1
        return out.view(-1, 1, n, n)
    
  