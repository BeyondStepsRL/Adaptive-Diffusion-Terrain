import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms


def transform_depth_image(image):
    image = image.unsqueeze(1).repeat(1, 3, 1, 1)
    image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
    return image
    

class TeacherNet(nn.Module):
    def __init__(self, encoder_channel=113):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = torch.nn.Sequential(*(list(resnet.children())[:-2]))
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.decoder = Decoder(113, encoder_channel)

    def forward(self, x):
        obs = x[:, :14]
        plg = x[:, 14:]

        # Reshape back to [N, 113, 3, 3]
        plg = plg.view(plg.size(0), 113, 3, 3)

        len_g = obs.size(dim=0)
        len_e = plg.size(dim=0)
        if len_g > len_e:
            y = self.decoder(plg.repeat(len_g // len_e, 1, 1, 1), obs)
        else:
            y = self.decoder(plg, obs)
        return y


class Decoder(nn.Module):
    def __init__(self, in_channels, goal_channels):
        super().__init__()
        self.relu    = nn.ReLU(inplace=True)
        self.tanh    = nn.Tanh()
        self.fg      = nn.Linear(14, goal_channels)

        self.conv1 = nn.Conv2d((in_channels + goal_channels), 256, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=2, stride=1, padding=0)

        self.fc1   = nn.Linear(1152, 256)
        self.fc2   = nn.Linear(256,  128)
        self.fc3   = nn.Linear(128,  2)

        self.frc1 = nn.Linear(256, 128)
        self.frc2 = nn.Linear(128, 1)

    def forward(self, plg, obs):
        # compute obs encoding
        obs = self.fg(obs[:, 0:14])
        obs = obs[:, :, None, None].expand(-1, -1, plg.shape[2], plg.shape[3])
        # cat x with obs in channel dim
        x = torch.cat((plg, obs), dim=1)
        # compute x
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = torch.flatten(x, 1)

        f = self.relu(self.fc1(x))

        a = self.relu(self.fc2(f))
        a = self.tanh(self.fc3(a))

        v = self.tanh(self.frc1(f))
        v = self.frc2(v)

        return a, v