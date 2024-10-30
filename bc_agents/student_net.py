import torch.nn as nn
import torch
import numpy as np
import torchvision.transforms as transforms
from ncps.torch import CfC
from ncps.wirings import AutoNCP
import torchvision.models as models


def transform_depth_image(image):
    # image = transforms.ToTensor()(image)
    resize = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST)
    image = resize(image.unsqueeze(1))
    image = image.expand(-1, 3, 224, 224)
    image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
    return image

# https://github.com/leggedrobotics/iPlanner
class EarlyStopScheduler(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                    verbose=False, threshold=1e-4, threshold_mode='rel',
                    cooldown=0, min_lr=0, eps=1e-8):
        super().__init__(optimizer=optimizer, mode=mode, factor=factor, patience=patience,
                            threshold=threshold, threshold_mode=threshold_mode,
                            cooldown=cooldown, min_lr=min_lr, eps=eps, verbose=verbose)
        self.no_decrease = 0

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            return self._reduce_lr(epoch)

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print('Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch, i, new_lr))
                return False
            else:
                return True

class LearnerNet(nn.Module):
    def __init__(self, encoder_channel=256, 
                 action_lower=torch.tensor([-2.0, -3.0], device=torch.device('cuda')),
                 action_upper=torch.tensor([2.0, 3.0], device=torch.device('cuda'))):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.encoder = resnet
        
        self.decoder = Decoder(250, encoder_channel, 2, action_lower, action_upper)

    def forward(self, obs, state, hx=None):
        obs = self.encoder(obs)
        obs = obs.view(obs.size(0), 250, 2, 2)
        x, hx = self.decoder(obs, state, hx)
        return x, hx


class Decoder(nn.Module):
    def __init__(self, in_channels, goal_channels, action_size, action_lower, action_upper):
        super().__init__()
        self.action_lower = action_lower
        self.action_upper = action_upper
        self.a = action_size

        self.relu    = nn.ReLU(inplace=True)
        self.tanh    = nn.Tanh()
        self.fg      = nn.Linear(14, goal_channels)

        self.conv1 = nn.Conv2d((in_channels + goal_channels), 256, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=2, stride=1, padding=0)

        self.fc1   = nn.Linear(512, 256)
        self.fc2   = nn.Linear(256,  128)
        self.rnn   = CfC(128, 64, batch_first=True, proj_size=self.a)

    def forward(self, obs, state, hx=None):
        # compute goal encoding
        state = self.fg(state[:, 0:14])
        state = state[:, :, None, None].expand(-1, -1, obs.shape[2], obs.shape[3])
        # cat x with goal in channel dim
        x = torch.cat((obs, state), dim=1)
        # compute x
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = torch.flatten(x, 1)

        f = self.relu(self.fc1(x))

        x = self.relu(self.fc2(f))
        x, hx = self.rnn(x, hx)
        x = self.tanh(x)

        x = (self.action_upper - self.action_lower) * (x + 1.0) / 2.0 + self.action_lower

        return x, hx