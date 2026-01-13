import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_activation(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.01)
    if name == "tanh":
        return nn.Tanh()
    if name == "elu":
        return nn.ELU()
    return nn.ReLU()


class MLP_base(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim1: int = 128,
        hidden_dim2: int = 64,
        dropout_rate: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()

        act1 = _make_activation(activation)
        act2 = _make_activation(activation)

        layers = [
            nn.Linear(input_dim, hidden_dim1),
            act1,
        ]
        if dropout_rate and dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        layers += [
            nn.Linear(hidden_dim1, hidden_dim2),
            act2,
        ]
        if dropout_rate and dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        layers += [
            nn.Linear(hidden_dim2, 1),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLP_enhance(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout_rate: float = 0.3,
        activation: str = "relu",
    ):
        super().__init__()

        act1 = _make_activation(activation)
        act2 = _make_activation(activation)

        h2 = max(int(hidden_dim // 2), 1)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            act1,
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, h2),
            nn.BatchNorm1d(h2),
            act2,
            nn.Dropout(dropout_rate),

            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.view_as(inputs).float()
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce)
        loss = self.alpha * (1.0 - pt) ** self.gamma * bce
        return loss.mean()


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        self.act1 = _make_activation(activation)
        self.act2 = _make_activation(activation)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.drop1 = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.drop2 = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.drop2(out)

        out = out + x
        out = self.act2(out)
        return out


class MLP_advanced(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        dropout_rate: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()

        self.act_in = _make_activation(activation)
        self.act_out1 = _make_activation(activation)

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            self.act_in,
        )

        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout=dropout_rate, activation=activation) for _ in range(int(num_blocks))]
        )

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            self.act_out1,
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        for block in self.blocks:
            out = block(out)
        return self.output_layer(out)