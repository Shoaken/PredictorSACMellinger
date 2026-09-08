"""Twin Q-functions used by SAC.

DoubleQCritic
    Two independent MLPs Q1(s,a), Q2(s,a). Used by:
      - SACAgent (library default; not a paper CLI experiment)
      - SPEDERAgentDomainRandomization (paper DR baseline)

CriticwithPhi
    Shared feature head phi_Q(s,a) then two linear heads.
    get_feature() is phi_Q in the paper. Used by:
      - Predictor (aligned to the nHSIC Predictor via lambda * nHSIC)
      - Vanilla SAC / STEADY stage 1 (spederv3)
      - STEADY stage 2 freezes this phi_Q as phi^circ

The two architectures are *not* interchangeable for reproducing the
tables: DR was trained with DoubleQCritic; Predictor/Vanilla SAC with
CriticwithPhi.

Adapted from https://github.com/denisyarats/pytorch_sac
"""

import torch
from torch import nn
import torch.nn.functional as F

from train.utils import util


class DoubleQCritic(nn.Module):
    """Independent twin Q-networks (Haarnoja SAC / pytorch_sac)."""

    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, device='cpu'):
        super().__init__()

        self.device = device

        self.Q1 = util.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = util.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.to(self.device)

        self.apply(util.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        return q1, q2


class CriticwithPhi(nn.Module):
    """Twin Q sharing a tanh feature phi_Q(s, a) in R^{feature_dim}.

    Architecture: Linear-ReLU x2 -> Linear-tanh (feature_dim) -> two Linear heads.
    Paper default feature_dim=512, hidden_dim=256.
    """

    def __init__(
            self,
            input_dim,
            feature_dim,
            hidden_dim=256,
            device='cuda'
    ):
        super().__init__()
        self.device = torch.device(device)
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, feature_dim)
        self.final_l1 = nn.Linear(feature_dim, 1)
        self.final_l2 = nn.Linear(feature_dim, 1)

        self.to(self.device)

        self.apply(util.weight_init)

    def get_feature(self, state, action):
        """Paper phi_Q(s, a). Used by Predictor alignment and STEADY skills."""
        current_device = next(self.parameters()).device

        state = state.to(current_device)
        action = action.to(current_device)

        x = torch.cat([state, action], axis=-1)
        f = F.relu(self.l1(x))
        f = F.relu(self.l2(f))
        f = F.tanh(self.l3(f))

        return f

    def forward(self, state, action):
        f = self.get_feature(state, action)
        q1 = self.final_l1(f)
        q2 = self.final_l2(f)
        return q1, q2
