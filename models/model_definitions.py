import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================================================================================
# [1] Baseline 모델 (BasicMLP)
# ========================================================================================
class BasicMLP(nn.Module):
    """
    [Baseline 모델]
    성능 비교의 기준(Baseline)을 수립하기 위한 가장 간단한 3-Layer MLP 구조입니다.
    사용처: MLP_base.ipynb

    구조 (Structure):
        Input -> Linear(128) -> ReLU -> Linear(128) -> ReLU -> Linear(1)

    인자 (Args):
        input_dim (int): 입력 피처의 개수
        hidden_dim (int): 은닉층(Hidden Layer)의 노드 수 (기본값: 128)
    """

    def __init__(self, input_dim, hidden_dim=128):
        super(BasicMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


# ========================================================================================
# [2] Enhanced 모델 (MLP_enhance) - 표준 블록
# ========================================================================================
class MLP_enhance(nn.Module):
    """
    [Enhanced 모델]
    Batch Normalization과 Dropout을 적용하여 학습 안정성과 일반화 성능을 높인 표준 MLP 모델입니다.
    용도: MLP_enhance.ipynb (단일 모델), MLP_advanced.ipynb (앙상블 구성원)

    구조 (Structure):
        Input -> [Linear -> BN -> Act -> Dropout] x 2 -> Output

    인자 (Args):
        input_dim (int): 입력 피처의 개수
        hidden_dim (int): 첫 번째 은닉층의 노드 수 (기본값: 128)
        dropout_rate (float): 드롭아웃 확률 (기본값: 0.3)
        activation (str): 활성화 함수 이름 ('relu', 'leaky_relu', 'elu', 'selu', 'tanh')
    """

    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.3, activation="relu"):
        super(MLP_enhance, self).__init__()

        # 활성화 함수 선택 (Activation Selection)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.01)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "selu":
            self.activation = nn.SELU()
        else:
            self.activation = nn.ReLU()

        self.net = nn.Sequential(
            # [1] 첫 번째 은닉층 (First Hidden Layer)
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 배치 정규화
            self.activation,
            nn.Dropout(dropout_rate),  # 과적합 방지
            # [2] 두 번째 은닉층 (Second Hidden Layer)
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            self.activation,
            nn.Dropout(dropout_rate),
            # [3] 출력층 (Output Layer)
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x)
