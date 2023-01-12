import sys
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class MakeConfig:
    def __init__(self, config):
        self.__dict__ = config

class Normal(nn.Module):
    def __init__(self, config, device):
        super(Normal, self).__init__()
        self.device = device
        self.config = config

    def sample(self):
        return torch.rand(1, self.config.index_dim, self.config.representation_dim, self.config.representation_dim).to(self.device) * self.config.num_levels
    
    def interpolate(self, X, Y):
        return (X + Y) / 2
    
    def reconstruct(self, X):
        return X

    def forward(self, X):
        return torch.rand(X.shape[0], self.num_levels, self.config.index_dim, X.shape[2], X.shape[3]).to(self.device)

def load_from_checkpoint(model, checkpoint_location):
    if os.path.exists(checkpoint_location):
        state_dict = torch.load(checkpoint_location, map_location=model.device)
        model.load_state_dict(state_dict)
    else:
        model.from_pretrained()
    return model

def get_lr(iter, learning_rate, config):
    if iter < config.warmup_iters:
        return learning_rate * iter / config.warmup_iters

    if iter > config.learning_rate_decay_iters:
        return config.min_learning_rate

    decay_ratio = (iter - config.warmup_iters) / (config.learning_rate_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    config.coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_learning_rate + config.coeff * (learning_rate - config.min_learning_rate)

        



