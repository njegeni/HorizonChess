"""
Supervised training loss for the chess network.

Combines the three heads' objectives:

    L = L_policy + value_weight * L_value + L_aux

where the auxiliary multi-step lookahead loss is

    L_aux = - sum_{k=1..n}  gamma^k * w_{p(k)} * sum_m pi(m | s_{t+k}) log pi_hat(m | s_t, k)

    * k          : horizon step (1..n plies ahead)
    * gamma^k    : discount -- further-out predictions are noisier, weight them less
    * w_{p(k)}   : per-parity weight. p(k) alternates opponent / self:
                   odd k  (t+1, t+3, ...) -> opponent's move
                   even k (t+2, t+4, ...) -> self's move
    * pi         : the REAL move played at s_{t+k} (from the game record)
    * pi_hat     : the network's prediction, made from s_t and k only

The aux term is a representation-shaping regulariser (KataGo keeps its
single-step opponent term at a small weight ~0.15); it is not the primary
objective, so w_self / w_opp are small.
"""

import torch
import torch.nn.functional as F


def _policy_loss(logits: torch.Tensor, target: torch.Tensor,
                 label_smoothing: float = 0.0) -> torch.Tensor:
    """Cross-entropy that accepts either hard labels (move indices, shape (B,))
    or soft labels (a distribution over moves, shape (B, 4672)).

    Soft labels are the book-position case: for opening positions that recur
    across many games you can build an empirical move-frequency distribution
    (from the recurrent-FEN table) instead of a one-hot played move, so the net
    isn't penalised for choosing a different-but-equally-valid book move."""
    if target.dim() == 1:                       # hard labels
        return F.cross_entropy(logits, target, label_smoothing=label_smoothing)
    logp = F.log_softmax(logits, dim=1)         # soft labels (distribution)
    return -(target * logp).sum(dim=1).mean()


def compute_loss(
    outputs: dict,
    targets: dict,
    *,
    value_weight: float = 1.0,
    gamma: float = 0.85,
    w_opp: float = 0.15,
    w_self: float = 0.15,
    label_smoothing: float = 0.0,
):
    """
    outputs  : dict from ChessNet.forward -> "policy" (B,4672), "value" (B,),
               "lookahead" (B, n, 4672)
    targets  : dict with
                 "policy"          : (B,) indices  OR  (B, 4672) distribution
                 "value"           : (B,) floats in [-1, 1]
                 "lookahead"       : (B, n) move indices for plies t+1..t+n
                 "lookahead_mask"  : (B, n) bool -- False where the game ended
                                     before that ply (target undefined)

    Returns (total_loss, parts_dict) where parts_dict holds detached scalars
    for logging.
    """
    # --- main policy loss ---
    policy_loss = _policy_loss(outputs["policy"], targets["policy"], label_smoothing)

    # --- value loss (MSE against game outcome) ---
    value_loss = F.mse_loss(outputs["value"], targets["value"])

    total = policy_loss + value_weight * value_loss

    # --- multi-step lookahead auxiliary loss ---
    aux_loss = outputs["policy"].new_zeros(())    # scalar tensor on the right device
    if "lookahead" in outputs:
        la_logits = outputs["lookahead"]          # (B, n, 4672)
        la_targets = targets["lookahead"]         # (B, n)
        la_mask = targets["lookahead_mask"]       # (B, n) bool
        n = la_logits.shape[1]

        for k in range(n):                        # k is 0-indexed; horizon step = k + 1
            step = k + 1
            valid = la_mask[:, k]
            if valid.sum() == 0:
                continue                          # whole batch ended before this ply
            ce = F.cross_entropy(
                la_logits[:, k][valid],
                la_targets[:, k][valid],
                label_smoothing=label_smoothing,
            )
            w = w_opp if step % 2 == 1 else w_self   # odd step = opponent, even = self
            aux_loss = aux_loss + (gamma ** step) * w * ce

        total = total + aux_loss

    parts = {
        "policy": policy_loss.detach(),
        "value": value_loss.detach(),
        "aux": aux_loss.detach(),
        "total": total.detach(),
    }
    return total, parts
