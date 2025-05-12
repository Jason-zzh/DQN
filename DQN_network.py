import torch
import torch.nn as nn
import torch.nn.functional as F

# Linear Q-Network (no conv) pass the test
class LinearModel(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.input_shape = input_shape
        c, h, w = input_shape
        self.fc = nn.Linear(c * h * w, num_actions)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Simple ConvNet pass the test
class SimpleConvNet(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.input_shape = input_shape
        c, h, w = input_shape
        self.net = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self._get_conv_out(input_shape), 256), nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def _get_conv_out(self, shape):
        c, h, w = shape
        # Calculate conv1 output
        h = (h - 8) // 4 + 1
        w = (w - 8) // 4 + 1
        # Calculate conv2 output
        h = (h - 4) // 2 + 1
        w = (w - 4) // 2 + 1
        return 32 * h * w

    def forward(self, x):
        # Ensure input is in NCHW format
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        if x.shape[-1] in [1, 3, 4]:  # If channels are last (NHWC)na
            x = x.permute(0, 3, 1, 2)  # Convert to NCHW
        return self.net(x)

# ConvNet with BatchNorm pass
class ConvNetBN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.input_shape = input_shape
        c, h, w = input_shape
        self.net = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=8, stride=4), nn.ReLU(), nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Flatten(),
            nn.Linear(self._get_conv_out(input_shape), 256), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Linear(256, num_actions)
        )

    def _get_conv_out(self, shape):
        return int(torch.flatten(self.net[:6](torch.zeros(1, *shape))).size(1))

    def forward(self, x):
        # Ensure input is in NCHW format
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        if x.shape[-1] in [1, 3, 4]:  # If channels are last (NHWC)
            x = x.permute(0, 3, 1, 2)  # Convert to NCHW
        return self.net(x)

# Simpler ConvNet pass
class TinyConvNet(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.input_shape = input_shape
        c, h, w = input_shape
        self.net = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self._get_conv_out(input_shape), 32), nn.ReLU(),
            nn.Linear(32, num_actions)
        )

    def _get_conv_out(self, shape):
        return int(torch.flatten(self.net[:4](torch.zeros(1, *shape))).size(1))

    def forward(self, x):
        return self.net(x)

# Nature DQN ConvNet pass
class NatureConvNetSimpler(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.input_shape = input_shape
        c, h, w = input_shape
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_size, 512), nn.ReLU(),
            nn.Linear(512, num_actions))
    
    def _get_conv_out(self, shape):
        c, h, w = shape
        
        # Conv1: 32x8x8, stride=4
        h = (h - 8) // 4 + 1
        w = (w - 8) // 4 + 1
        
        # Conv2: 64x4x4, stride=2
        h = (h - 4) // 2 + 1
        w = (w - 4) // 2 + 1
        
        # Conv3: 64x3x3, stride=1
        h = (h - 3) // 1 + 1
        w = (w - 3) // 1 + 1
        
        return 64 * h * w
    
    def forward(self, x):
        # Ensure input is in NCHW format
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        if x.shape[-1] in [1, 3, 4]:  # If channels are last (NHWC)
            x = x.permute(0, 3, 1, 2)  # Convert to NCHW
        conv_out = self.conv_layers(x)
        return self.fc_layers(conv_out)

# Dueling DQN ConvNet
class DuelingConvNet(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.input_shape = input_shape
        c, h, w = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),
            nn.Flatten()
        )
        conv_out_size = self._get_conv_out(input_shape)
        
        self.adv_fc1 = nn.Linear(conv_out_size, 256)
        self.adv_fc2 = nn.Linear(256, num_actions)
        
        self.val_fc1 = nn.Linear(conv_out_size, 256)
        self.val_fc2 = nn.Linear(256, 1)

    def _get_conv_out(self, shape):
        return int(self.conv(torch.zeros(1, *shape)).size(1))

    def forward(self, x):
        conv_out = self.conv(x)
        
        adv = F.relu(self.adv_fc1(conv_out))
        adv = self.adv_fc2(adv)
        adv = adv - adv.mean(1, keepdim=True)  # A - mean(A)
        
        val = F.relu(self.val_fc1(conv_out))
        val = self.val_fc2(val)
        val = val.expand(-1, adv.size(1))  # 显式广播
        
        return val + adv

# Optional: factory function
def get_model(name, input_shape, num_actions):
    models = {
        'linear': LinearModel,
        'convnet': SimpleConvNet,
        'convnet_bn': ConvNetBN,
        'simpler_convnet': TinyConvNet,
        'nature_convnet': NatureConvNetSimpler,
        'dueling_convnet': DuelingConvNet
    }
    if name not in models:
        raise ValueError(f"Unknown model name: {name}")
    return models[name](input_shape, num_actions)
