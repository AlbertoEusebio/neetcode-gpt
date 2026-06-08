import torch
import numpy as np
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        means = torch.zeros(fan_out, fan_in)
        std = torch.ones(fan_out, fan_in) * (2 / (fan_in + fan_out))**0.5

        matrix = torch.normal(mean=means, std=std)
        # print(matrix)
        return matrix.tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        means = torch.zeros(fan_out, fan_in)
        std = torch.ones(fan_out, fan_in) * (2 / (fan_in))**0.5

        matrix = torch.normal(mean=means, std=std)
        # print(matrix)
        return matrix.tolist()
    
    def random_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        means = torch.zeros(fan_out, fan_in)
        std = torch.ones(fan_out, fan_in)

        matrix = torch.normal(mean=means, std=std)
        # print(matrix)
        return matrix.tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        torch.manual_seed(0)

        stds = []
        prev_d = input_dim
        weights = []
        for i in range(num_layers):
            if init_type == 'xavier':
                st = math.sqrt(2.0 / (prev_d+hidden_dim))
            elif init_type == 'kaiming':
                st = math.sqrt(2.0 / (prev_d))
            elif init_type == 'random':
                st = 1
            
            W = torch.randn(hidden_dim, prev_d) * st
            weights.append(W)
            prev_d = hidden_dim


        x = torch.randn(1, input_dim)
        for W in weights:
            x = x @ W.T
            x = torch.relu(x)
            stds.append(round(x.std().item(), 2))


        return stds