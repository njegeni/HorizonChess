import torch
import torch.nn.functional as F

def compute_loss(outputs, targets, value_weight=1.0, gamma=0.85, w_opp=0.15, w_self=0.15):
    policy_loss = F.cross_entropy(outputs["policy"], targets["policy"])
    value_loss = F.mse_loss(outputs["value"], targets["value"])
    total = policy_loss + value_weight * value_loss

    # multi-step lookahead aux loss:
    #   sum_{k=1..n} gamma^k * w_p(k) * cross_entropy(pred_k, real_move_{t+k})
    # p(k): odd horizon = opponent's move (w_opp), even horizon = self (w_self).
    n = outputs["lookahead"].shape[1]
    aux_loss = outputs["policy"].new_zeros(())        # scalar on the right device
    for k in range(n):                                # k is 0-indexed; horizon = k+1
        valid = targets["lookahead_mask"][:, k]
        if valid.sum() == 0:
            continue                                  # game ended before this ply
        ce = F.cross_entropy(
            outputs["lookahead"][:, k][valid],        # (valid, 4672)
            targets["lookahead"][:, k][valid],        # (valid,)
        )
        step = k + 1
        w = w_opp if step % 2 == 1 else w_self        # odd = opponent, even = self
        aux_loss = aux_loss + (gamma ** step) * w * ce
    total = total + aux_loss

    # detached scalars for logging
    parts = {
        "policy": policy_loss.detach(),
        "value": value_loss.detach(),
        "aux": aux_loss.detach(),
        "total": total.detach(),
    }
    return total, parts


if __name__ == "__main__":
    torch.manual_seed(0)
    B, n = 4, 2
    outputs = {
        "policy": torch.randn(B, 4672, requires_grad=True),
        "value": torch.tanh(torch.randn(B, requires_grad=True)),
        "lookahead": torch.randn(B, n, 4672, requires_grad=True),
    }
    mask = torch.ones(B, n, dtype=torch.bool)
    mask[0, -1] = False                               # one position ends early
    targets = {
        "policy": torch.randint(0, 4672, (B,)),
        "value": torch.empty(B).uniform_(-1, 1),
        "lookahead": torch.randint(0, 4672, (B, n)),
        "lookahead_mask": mask,
    }
    total, parts = compute_loss(outputs, targets)
    total.backward()
    print({k: round(v.item(), 4) for k, v in parts.items()})
    print("total.requires_grad:", total.requires_grad)
