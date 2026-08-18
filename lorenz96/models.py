import torch
import torch.nn as nn


class CircularCNN(nn.Module):
    # Circular padding + weight sharing match the L96 ring and its site-invariant 4-wide stencil.
    def __init__(self, hidden=64, n_layers=3, kernel=5, in_channels=1, heteroscedastic=False):
        super().__init__()
        self.in_channels = in_channels
        self.heteroscedastic = heteroscedastic
        pad = kernel // 2

        def conv(cin, cout):
            return nn.Conv1d(cin, cout, kernel, padding=pad, padding_mode="circular")

        layers, c = [], in_channels
        for _ in range(n_layers):
            layers += [conv(c, hidden), nn.SiLU()]
            c = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv1d(hidden, 2 if heteroscedastic else 1, kernel_size=1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out = self.head(self.body(x))
        if self.heteroscedastic:
            mean, log_var = out[:, 0], out[:, 1]
            return mean, log_var.clamp(-14.0, 6.0)
        return out.squeeze(1)


class Persistence(nn.Module):
    # Baseline: predict zero tendency, i.e. the state does not change.
    def forward(self, x):
        if x.dim() == 3:
            x = x[:, -1]
        return torch.zeros_like(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
