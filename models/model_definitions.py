import torch
import torch.nn as nn
import torch.nn.functional as F


# --- MLP_enhance (from EnhancedMLP) ---
class MLP_enhance(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.3, activation="relu"):
        super(MLP_enhance, self).__init__()

        # 활성화 함수 선택 (Activation Function Selection)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.01)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()  # 기본값

        self.net = nn.Sequential(
            # [Layer 1]
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # [안정화]
            self.activation,
            nn.Dropout(dropout_rate),  # [과적합 방지]
            # [Layer 2]
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            self.activation,
            nn.Dropout(dropout_rate),
            # [Output]
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


# --- Components for MLP_advanced ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return torch.mean(F_loss)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1, activation="relu"):
        super(ResidualBlock, self).__init__()

        # 활성화 함수 선택
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "leaky_relu":
            self.act = nn.LeakyReLU(0.01)
        elif activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "elu":
            self.act = nn.ELU()
        else:
            self.act = nn.ReLU()

        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        return self.act(out + residual)


class MLP_advanced(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        num_blocks=2,
        dropout_rate=0.1,
        activation="relu",
    ):
        super(MLP_advanced, self).__init__()

        # 활성화 함수 선택
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "leaky_relu":
            self.act = nn.LeakyReLU(0.01)
        elif activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "elu":
            self.act = nn.ELU()
        else:
            self.act = nn.ReLU()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), self.act
        )
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim, dropout=dropout_rate, activation=activation)
                for _ in range(num_blocks)
            ]
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 32), self.act, nn.Linear(32, 1)
        )

    def forward(self, x):
        out = self.input_layer(x)
        for block in self.blocks:
            out = block(out)
        return self.output_layer(out)
