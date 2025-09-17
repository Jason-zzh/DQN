# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils import clip_grad_norm_

from model import DQN


class Agent():
  def __init__(self, args, env):
    # Convert fixed parameters to learnable parameters
    self.eer = nn.Parameter(torch.tensor(4.0))  # Initial value 4.0
    self.errd = nn.Parameter(torch.tensor(3.0))  # Initial value 3.0
    self.e2 = nn.Parameter(torch.tensor(7.0))  # Initial value 7.0
    self.pr = nn.Parameter(torch.tensor(1.0))  # Initial value 1.0
    
    self.dc=0.0
    self.de_rate=1.0
    self.cover=1.0
    self.action_space = env.action_space()
    self.atoms = args.atoms
    self.Vmin = args.V_min
    self.Vmax = args.V_max
    self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)  # Support (range) of z
    self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
    self.batch_size = args.batch_size
    self.n = args.multi_step
    self.discount = args.discount
    self.norm_clip = args.norm_clip

    self.online_net = DQN(args, self.action_space).to(device=args.device)
    if args.model:  # Load pretrained model if provided
      if os.path.isfile(args.model):
        state_dict = torch.load(args.model, map_location='cpu')
        if 'conv1.weight' in state_dict.keys():
          for old_key, new_key in (('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'), ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
            state_dict[new_key] = state_dict[old_key]
            del state_dict[old_key]
        self.online_net.load_state_dict(state_dict)
        print("Loading pretrained model: " + args.model)
      else:
        raise FileNotFoundError(args.model)

    self.online_net.train()

    self.target_net = DQN(args, self.action_space).to(device=args.device)
    self.update_target_net()
    self.target_net.train()
    for param in self.target_net.parameters():
      param.requires_grad = False

    self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

  def _process_weights_fft(self, model):
    """Process model weights with FFT"""
    for name, param in model.named_parameters():
      if 'weight' in name and 'fc' in name:  # Only process NoisyLinear weights
        # Get weight tensor
        w = param
        fft_w = torch.fft.fft2(w)
        w_fd = fft_w.clone()
        
        xmax, ymax = w.shape
        x = torch.arange(0, xmax, device=w.device)
        y = torch.arange(0, ymax, device=w.device)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        # Calculate mx and my
        mx = 2 * torch.pi * xx / (xmax - 1)
        my = 2 * torch.pi * yy / (ymax - 1)
        
        # Calculate dx and dy using torch operations
        dx = (mx**self.eer) / ((1 + self.errd * mx**2)**self.e2)
        dy = (my**self.eer) / ((1 + self.errd * my**2)**self.e2)
        mm = self.pr * (dx + dy)
        
        # Set DC component
        mm[0, 0] = self.dc
        
        # Apply modulation
        w_fd = w_fd * (mm + self.cover)
        
        # Update pr parameter value
        # with torch.no_grad():
        # self.pr.data *= self.de_rate
        
        # Inverse FFT and clip
        w_out = torch.fft.ifft2(w_fd).real
        w_min = torch.min(w)
        w_max = torch.max(w)
        w_out = torch.clamp(w_out, w_min, w_max)
        
        # Update parameter
        param.data.copy_(w_out)

  # ... (rest of the class methods remain the same as in original agent.py)
