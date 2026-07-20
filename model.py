import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

board = 8
policy_planes = 73
policy_size = policy_planes * board * board


@dataclass
class ModelConfig:
    in_planes: int = 102
    channels: int = 128
    num_blocks: int = 10
    lookahead_n: int = 2
    value_hidden: int = 256

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        skip = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + skip)


class PolicyHead(nn.Module):
    #logits over the 73x8x8 move encoding (4672). flatten stays in
    #(plane, row, col) order so it matches encode_move.
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.head = nn.Conv2d(channels, policy_planes, 1)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.head(x)            # (B, 73, 8, 8)
        return x.flatten(1)         # (B, 4672)


class ValueHead(nn.Module):
    #scalar outcome in [-1, 1] from the side-to-move's perspective
    def __init__(self, channels, hidden):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(board * board, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.flatten(1)            # (B, 64)
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)).squeeze(1)   # (B,)


class LookaheadHead(nn.Module):
    #aux head: predicts the move distribution expected k plies ahead, from the
    #current features + a learned embedding of k. one head serves all horizons.
    def __init__(self, channels, horizon):
        super().__init__()
        self.horizon = horizon
        self.k_embed = nn.Embedding(horizon, channels)
        self.proj = nn.Linear(2 * channels, channels)
        self.out = nn.Linear(channels, policy_size)

    def forward(self, x, k_index):
        mean = x.mean(dim=(2, 3))               # (B, C)
        mx = x.amax(dim=(2, 3))                 # (B, C)
        h = F.relu(self.proj(torch.cat([mean, mx], dim=1)))
        h = h + self.k_embed(k_index)           # condition on which horizon
        return self.out(h)                      # (B, 4672)


class ChessNet(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.config = config
        self.stem_conv = nn.Conv2d(config.in_planes, config.channels, 3, padding=1)
        self.stem_bn = nn.BatchNorm2d(config.channels)
        self.blocks = nn.ModuleList(
            [ResidualBlock(config.channels) for _ in range(config.num_blocks)]
        )
        self.policy_head = PolicyHead(config.channels)
        self.value_head = ValueHead(config.channels, config.value_hidden)
        self.lookahead_head = LookaheadHead(config.channels, config.lookahead_n)

    def trunk(self, x):
        x = F.relu(self.stem_bn(self.stem_conv(x)))
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, x):
        feats = self.trunk(x)
        n = self.config.lookahead_n
        b = x.shape[0]
        steps = []
        for k in range(n):
            k_idx = torch.full((b,), k, dtype=torch.long, device=x.device)
            steps.append(self.lookahead_head(feats, k_idx))
        return {
            "policy": self.policy_head(feats),      # (B, 4672)
            "value": self.value_head(feats),        # (B,)
            "lookahead": torch.stack(steps, dim=1), # (B, n, 4672)
        }


if __name__ == "__main__":
    net = ChessNet()
    n_params = sum(p.numel() for p in net.parameters())
    print(f"parameters: {n_params:,}")

    dummy = torch.randn(4, 102, board, board)
    out = net(dummy)
    print("policy   :", tuple(out["policy"].shape))
    print("value    :", tuple(out["value"].shape))
    print("lookahead:", tuple(out["lookahead"].shape))
