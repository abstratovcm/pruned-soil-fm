import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2
        self.emb = np.log(10000) / (self.half_dim - 1)

    def forward(self, t):
        # t is (batch_size, 1) or (batch_size,)
        if t.dim() == 1:
            t = t.unsqueeze(1)

        emb = torch.exp(torch.arange(self.half_dim, device=t.device) * -self.emb)
        emb = t * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        h = self.linear1(h)
        h = self.act1(h)
        h = self.drop1(h)

        h = self.norm2(h)
        h = self.linear2(h)
        h = self.act2(h)
        h = self.drop2(h)

        return x + h

import os
import json

class VelocityPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, time_emb_dim=32, dropout=0.1, num_blocks=4):
        super().__init__()
        self.input_dim = input_dim
        self.time_emb_dim = time_emb_dim
        self.num_blocks = num_blocks

        # Embeddings
        self.time_embed = TimeEmbedding(time_emb_dim)

        # ResNet Architecture
        input_total_dim = self.input_dim + time_emb_dim



        self.input_proj = nn.Sequential(
            nn.Linear(input_total_dim, hidden_dim),
            nn.GELU()
        )

        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(hidden_dim, dropout))
        self.blocks = nn.Sequential(*blocks)

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.input_dim)
        )

    def forward(self, x, t):
        """
        Predicts the vector field v(x, t).
        x: (B, D)
        t: (B,)
        """
        t_emb = self.time_embed(t)
        h = torch.cat([x, t_emb], dim=1)

        h = self.input_proj(h)
        h = self.blocks(h)
        return self.output_proj(h)

    def compute_loss(self, x1_full):
        """
        Computes the Exact OT-VelocityPredictor loss.
        x1_full: Real data sample (B, D_out)
        """
        b, d_out = x1_full.shape
        device = x1_full.device

        x0_base = torch.randn_like(x1_full)

        x0_cpu = x0_base.detach().cpu().numpy()
        x1_cpu = x1_full.detach().cpu().numpy()

        diff = x0_cpu[:, None, :] - x1_cpu[None, :, :]
        dist_matrix = np.sum(diff**2, axis=-1)

        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        x0_base_paired = x0_base[row_ind]
        x1_full_paired = x1_full[col_ind]

        t = torch.rand(b, device=device)
        t_reshaped = t.view(-1, 1)

        x_t_base = t_reshaped * x1_full_paired + (1 - t_reshaped) * x0_base_paired

        target_v_base = x1_full_paired - x0_base_paired

        pred_v_base = self.forward(x_t_base, t)

        loss = torch.mean((pred_v_base - target_v_base) ** 2)

        return loss

    @torch.no_grad()
    def sample(self, num_samples, steps=100, device='cpu', return_trajectory=False, seed=None):
        """
        Generates samples using ODE solver with Classifier-Free Guidance (CFG).
        """
        from torchdiffeq import odeint

        if seed is not None:
            torch.manual_seed(seed)

        # Initial condition x0 ~ N(0, I) STRICTLY for the input dimension (base features)
        x0_base = torch.randn(num_samples, self.input_dim, device=device)

        # Wrapper for ODE solver
        def ode_func(t, x_base):
            # t is scalar tensor
            t_batch = t.expand(x_base.shape[0])

            # Network input is exactly x_base
            v_full = self.forward(x_base, t_batch)

            return v_full

        t_span = torch.linspace(0, 1, steps, device=device)

        traj_base = odeint(ode_func, x0_base, t_span, method='euler')
        final_base = traj_base[-1]

        if return_trajectory:
            return traj_base, t_span
        return final_base
