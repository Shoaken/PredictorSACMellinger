"""Feature networks named in the paper.

MLPFeaturePhi_IB  Z = phi(s, a) for the Predictor nHSIC bottleneck.
                  Not the actor input and not the critic's phi_Q.

MLPFeatureMu      mu(s') next-state features for STEADY Hilbert-Schmidt
                  skill learning. Stage 1 (spederv3) trains this; stage 2
                  freezes it as mu^circ.

MLPFeaturePhi     residual phi(s, a) learned only in STEADY stage 2.
                  Concatenated with frozen phi^circ (critic get_feature).

Bounded arctan outputs keep features in a compact range for the
inner-product / nHSIC estimators. Paper default feature_dim=512.
STEADY residual width is TransferAgent.aug_feature_dim (default 128).
"""
import torch
from torch import nn
from torch.nn import functional as F


class MLPFeaturePhi(nn.Module):
    """STEADY residual skill phi(s, a). Used only by TransferAgent."""

    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim=256,
            feature_dim=256
    ):
        super(MLPFeaturePhi, self).__init__()

        self.feature_dim = feature_dim

        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, feature_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], axis=-1)
        z = F.elu(self.l1(x))
        z = F.elu(self.l2(z))
        logit = torch.arctan(self.l3(z))
        return logit


class MLPFeatureMu(nn.Module):
    """STEADY next-state skill mu(s'). Frozen as mu^circ in stage 2."""

    def __init__(
            self,
            state_dim,
            hidden_dim=256,
            feature_dim=256,
            device='cpu'
    ):
        super(MLPFeatureMu, self).__init__()
        self.device = torch.device(device)

        self.feature_dim = feature_dim

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, feature_dim)

        self.to(self.device)

    def forward(self, state):
        current_device = next(self.parameters()).device
        state = state.to(current_device)
        z = F.elu(self.l1(state))
        z = F.elu(self.l2(z))
        # 1/d scaling matches the Hilbert-Schmidt inner-product skill loss.
        logit = 1 / self.feature_dim * torch.arctan(self.l3(z))
        return logit


class MLPFeaturePhi_IB(nn.Module):
    """Predictor Z = phi(s, a) for L_IB = nHSIC(Z,[s,a]) - beta nHSIC(Z,s')."""

    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim=256,
            feature_dim=256,
            device='cuda'
    ):
        super(MLPFeaturePhi_IB, self).__init__()
        self.device = torch.device(device)
        self.feature_dim = feature_dim

        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, feature_dim)

        self.to(self.device)

    def forward(self, state, action):
        x = torch.cat([state, action], axis=-1)
        z1 = F.elu(self.l1(x))
        z2 = F.elu(self.l2(z1))
        return torch.arctan(self.l3(z2))
