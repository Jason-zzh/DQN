from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from model import DQN

class FFTmodule(nn.Module):
    def __init__(self,xmax,ymax):
        super().__init__()
        self.eer = nn.Parameter(torch.tensor(4.0))  
        self.errd = nn.Parameter(torch.tensor(3.0))
        self.e2 = nn.Parameter(torch.tensor(7.0))
        self.pr = nn.Parameter(torch.tensor(1.0))
        self.dc = 0.0
        self.cover = 1.0
        x = torch.arange(0, xmax)
        y = torch.arange(0, ymax)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        mx = 2 * torch.pi * xx / (xmax - 1)
        my = 2 * torch.pi * yy / (ymax - 1)
        dx = (mx**self.eer) / ((1 + self.errd * mx**2)**self.e2)
        dy = (my**self.eer) / ((1 + self.errd * my**2)**self.e2)
        self.mm = self.pr * (dx + dy)
        self.mm[0, 0] = self.dc
        
    def fft_processing(self,w):
        # w_min = torch.min(w)
        # w_max = torch.max(w)
        w = torch.fft.fft2(w)
        w = w * self.mm + w
        w = torch.fft.ifft2(w).real
        print("FFT梯度:", w.grad)
        # w = torch.clamp(w, w_min, w_max)
        return w
    
    def forward(self, input, T):
        if T >= 50 and T % 50 == 0:
            input=self.fft_processing(input)
        return input

class TwoLayerLinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        
        # 获取权重形状 (out_features, in_features)
        self.fft_module1 = FFTmodule(
            self.linear1.weight.shape[0], 
            self.linear1.weight.shape[1]
        )
        self.fft_module2 = FFTmodule(
            self.linear2.weight.shape[0], 
            self.linear2.weight.shape[1]
        )
        
    def forward(self, x, T=None):
        # 修改权重（安全方式）
        if T is not None:
            self.linear1.weight = self.fft_module1(
                self.linear1.weight, 
                T
            )
            self.linear2.weight = self.fft_module2(
                self.linear2.weight, 
                T
            )
        
        h = F.relu(self.linear1(x))
        return self.linear2(h)

def main():
    # Hyperparameters
    input_dim = 10
    hidden_dim = 10
    output_dim = 10
    learning_rate = 0.0001
    epochs = 1000
    batch_size = 32

    # Create model
    model = TwoLayerLinear(input_dim, hidden_dim, output_dim)
    model.train()
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Generate random data
    X = torch.randn(1000, input_dim)
    y = torch.randn(1000, output_dim)

    # Training loop
    for epoch in range(epochs):
        # Forward pass with FFT weight processing
        outputs = model(X, T=epoch)
        loss = criterion(outputs, y)
        
        # Print detailed gradient information
        if (epoch) % 50 == 0:
            print("\n--- FFT Parameter Gradients ---")
            for name, param in model.fft_module1.named_parameters():
                grad = param.grad
                print(f"{name}: value={param.item():.4f} | grad={'None' if grad is None else 'False detect'}")
            
            # Verify computation graph
            print("\n--- Gradient Flow Check ---")
            print(f"Linear1 weight grad: {'None' if model.linear1.weight.grad is None else model.linear1.weight.grad.norm().item()}")
            print(f"Linear2 weight grad: {'None' if model.linear2.weight.grad is None else model.linear2.weight.grad.norm().item()}")
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    print("Training complete!")

if __name__ == "__main__":
    main()
