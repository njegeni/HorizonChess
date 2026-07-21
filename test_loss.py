import math
import torch
from loss import compute_loss

N = 4672   # policy size


#build outputs/targets with uniform (zeros) logits, so every cross-entropy is
#exactly ln(N) -- lets us predict the aux loss in closed form.
def uniform_case(B=4, n=2, mask=None):
    outputs = {
        "policy": torch.zeros(B, N),
        "value": torch.zeros(B),
        "lookahead": torch.zeros(B, n, N),
    }
    if mask is None:
        mask = torch.ones(B, n, dtype=torch.bool)
    targets = {
        "policy": torch.randint(0, N, (B,)),
        "value": torch.zeros(B),
        "lookahead": torch.randint(0, N, (B, n)),
        "lookahead_mask": mask,
    }
    return outputs, targets


#uniform logits -> policy cross-entropy is the ln(N) baseline
def test_baseline_policy():
    outputs, targets = uniform_case()
    _, parts = compute_loss(outputs, targets)
    assert abs(parts["policy"].item() - math.log(N)) < 1e-3


#confident correct logits + exact value -> policy and value losses ~0
def test_perfect_prediction():
    B = 4
    idx = torch.randint(0, N, (B,))
    policy = torch.zeros(B, N)
    policy[torch.arange(B), idx] = 30.0        # nearly one-hot on the target
    val = torch.empty(B).uniform_(-1, 1)
    outputs = {"policy": policy, "value": val, "lookahead": torch.zeros(B, 2, N)}
    targets = {
        "policy": idx,
        "value": val.clone(),                  # prediction == target
        "lookahead": torch.randint(0, N, (B, 2)),
        "lookahead_mask": torch.zeros(B, 2, dtype=torch.bool),  # aux off
    }
    _, parts = compute_loss(outputs, targets)
    assert parts["policy"].item() < 1e-3
    assert parts["value"].item() == 0.0
    assert parts["aux"].item() == 0.0


#a fully-masked horizon contributes nothing and never NaNs
def test_aux_all_masked():
    outputs, targets = uniform_case(mask=torch.zeros(4, 2, dtype=torch.bool))
    total, parts = compute_loss(outputs, targets)
    assert parts["aux"].item() == 0.0
    assert torch.isfinite(total)


#aux = sum_k gamma^k * w * ce, with ce = ln(N) for uniform logits
def test_aux_gamma_and_weight():
    outputs, targets = uniform_case(n=2)
    _, parts = compute_loss(outputs, targets, gamma=0.85, w_opp=0.15, w_self=0.15)
    L = math.log(N)
    expected = 0.85**1 * 0.15 * L + 0.85**2 * 0.15 * L
    assert abs(parts["aux"].item() - expected) < 1e-3


#parity: odd horizon uses w_opp, even horizon uses w_self
def test_aux_parity_weights():
    outputs, targets = uniform_case(n=2)
    # only opponent (odd, horizon 1) counts; self weight zeroed
    _, parts = compute_loss(outputs, targets, gamma=1.0, w_opp=1.0, w_self=0.0)
    assert abs(parts["aux"].item() - math.log(N)) < 1e-3   # just horizon 1
    # now only self (even, horizon 2) counts
    _, parts = compute_loss(outputs, targets, gamma=1.0, w_opp=0.0, w_self=1.0)
    assert abs(parts["aux"].item() - math.log(N)) < 1e-3   # just horizon 2


#value_weight scales the value term in the total
def test_value_weight():
    outputs, targets = uniform_case(mask=torch.zeros(4, 2, dtype=torch.bool))
    outputs["value"] = torch.full((4,), 0.5)
    targets["value"] = torch.zeros(4)          # mse = 0.25
    _, parts = compute_loss(outputs, targets, value_weight=2.0)
    # total = policy + 2*value + aux(0)
    assert abs(parts["total"].item() - (parts["policy"].item() + 2 * 0.25)) < 1e-3


#label smoothing raises the loss floor: even a perfect prediction isn't 0
def test_label_smoothing():
    B = 4
    idx = torch.randint(0, N, (B,))
    policy = torch.zeros(B, N)
    policy[torch.arange(B), idx] = 30.0        # confident, correct
    outputs = {"policy": policy, "value": torch.zeros(B), "lookahead": torch.zeros(B, 2, N)}
    targets = {
        "policy": idx, "value": torch.zeros(B),
        "lookahead": torch.randint(0, N, (B, 2)),
        "lookahead_mask": torch.zeros(B, 2, dtype=torch.bool),
    }
    _, hard = compute_loss(outputs, targets, label_smoothing=0.0)
    _, soft = compute_loss(outputs, targets, label_smoothing=0.1)
    assert hard["policy"].item() < 1e-3          # perfect -> ~0 without smoothing
    assert soft["policy"].item() > 0.5           # smoothing keeps a floor


#total carries grad; the logged parts are detached
def test_grad_and_detached_parts():
    outputs, targets = uniform_case()
    outputs["policy"] = outputs["policy"].requires_grad_(True)
    outputs["lookahead"] = outputs["lookahead"].requires_grad_(True)
    total, parts = compute_loss(outputs, targets)
    assert total.requires_grad
    total.backward()
    for v in parts.values():
        assert not v.requires_grad
