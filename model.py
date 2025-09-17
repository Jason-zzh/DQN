from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor


# Custom NoisyLinear with FFT processing
class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.5, 
               eer=3.0, errd=2.0, e2=8.0, pr=1.0, dc=0.0, cover=1.0):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer('bias_epsilon', torch.empty(out_features))
    
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

  def _scale_noise(self, size):
    x = torch.randn(size, device=self.weight_mu.device)
    return x.sign().mul_(x.abs().sqrt_())

  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  def forward(self, input):
    if self.training:
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)

class MyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.eer = nn.Parameter(torch.tensor(3.0))          
        self.errd = nn.Parameter(torch.tensor(2.0))        
        self.e2 = nn.Parameter(torch.tensor(8.0))          
        self.pr = nn.Parameter(torch.tensor(0.1))       
        self.l = nn.Parameter(torch.tensor(0.0))
        
        xr = torch.arange(0, self.out_features, dtype=torch.float32).view(self.out_features, 1)
        yr = torch.arange(0, self.in_features, dtype=torch.float32).view(1, self.in_features)
        self.register_buffer('xr', xr)
        self.register_buffer('yr', yr)
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='linear')
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            nn.init.zeros_(self.bias)  
        else:
            self.register_parameter('bias', None)
            
        self._init_freq_mask()

    def _init_freq_mask(self):
        mx = (self.xr * 2 * math.pi / (self.out_features - 1)).to(self.device)
        my = (self.yr * 2 * math.pi / (self.in_features - 1)).to(self.device)
        
        self.register_buffer('mx', mx)
        self.register_buffer('my', my)

    def forward(self, x: Tensor) -> Tensor:
        weight_if = torch.fft.fft2(self.weight)
        dx = (self.mx ** self.eer) / ((1 + self.errd * (self.mx ** 2)) ** self.e2).to(self.device)
        dy = (self.my ** self.eer) / ((1 + self.errd * (self.my ** 2)) ** self.e2).to(self.device)
        mm = self.pr * (dx + dy).to(self.device)
        mm[0, 0] = self.l  
        weight_if = weight_if * (mm + 1)
        weight_it = torch.fft.ifft2(weight_if).real
        weight_it = torch.clamp(weight_it, -1, 1)
        
        return F.linear(x, weight_it, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
      
class DQN(nn.Module):
  def __init__(self, args, action_space):
    super(DQN, self).__init__()
    self.atoms = args.atoms
    self.action_space = action_space

    if args.architecture == 'canonical':
      self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 8, stride=4, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU())
      self.conv_output_size = 3136
    elif args.architecture == 'data-efficient':
      self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 5, stride=5, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 5, stride=5, padding=0), nn.ReLU())
      self.conv_output_size = 576
    self.fc_h_v = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_h_a = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
    self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)
    # self.fc_h_v = MyLinear(self.conv_output_size, args.hidden_size, 'cuda')
    # self.fc_h_a = MyLinear(self.conv_output_size, args.hidden_size, 'cuda')
    # self.fc_z_v = MyLinear(args.hidden_size, self.atoms, 'cuda')
    # self.fc_z_a = MyLinear(args.hidden_size, action_space * self.atoms, 'cuda')

  def forward(self, x, log=False):
    x = self.convs(x)
    x = x.view(-1, self.conv_output_size)
    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    if log:  # Use log softmax for numerical stability
      q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
    else:
      q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
    return q

  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()