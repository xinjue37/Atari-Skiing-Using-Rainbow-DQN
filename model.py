# -*- coding: utf-8 -*-
from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F

 
# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.5):
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
      # print(type(input), input.dtype)
      # print(type(self.weight_mu + self.weight_sigma * self.weight_epsilon))
      # print(type(self.bias_mu + self.bias_sigma * self.bias_epsilon))
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
        return F.linear(input, self.weight_mu, self.bias_mu)



class DQN(nn.Module):
  def __init__(self, args, action_space):
    super(DQN, self).__init__()
    self.atoms = args.atoms
    self.device = args.device
    self.action_space = action_space

    self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 7, stride=1, padding="same"), nn.ReLU(),
                               nn.MaxPool2d(kernel_size=2, stride=2),
                               nn.Conv2d(32, 64, 5, stride=1, padding="same"), nn.ReLU(),
                               nn.MaxPool2d(kernel_size=2, stride=2),
                               nn.Conv2d(64, 64, 5, stride=1, padding="same"), nn.ReLU(),
                               nn.MaxPool2d(kernel_size=2, stride=2),
                               nn.Conv2d(64, 32, 5, stride=1, padding="same"), nn.ReLU(),
                               nn.MaxPool2d(kernel_size=2, stride=2),
                               nn.Conv2d(32, 3, 5, stride=1, padding="same"), nn.ReLU())
    
    self.fc_input_size = 3*8*9

    self.fc_h_v = NoisyLinear(self.fc_input_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_h_a = NoisyLinear(self.fc_input_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
    self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)


  def forward(self, x, log=False):
    # Assuming x is a tensor of shape (batch_size, 2 * len(args.history_length))
    
    x = self.convs(x)
    
    x = x.view(-1, self.fc_input_size).to(device=self.device)
    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    if log:  # Use log softmax for numerical stability
        q = F.log_softmax(q, dim=2)  # Log probabilities with action over the second dimension
    else:
        q = F.softmax(q, dim=2)      # Probabilities with action over the second dimension
    return q


  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()
