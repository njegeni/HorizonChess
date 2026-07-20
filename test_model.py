import torch
from model import ChessNet, ModelConfig, ResidualBlock, policy_size


#small config keeps the tests fast
def small(**kw):
    return ModelConfig(channels=8, num_blocks=1, value_hidden=16, **kw)


#residual block preserves shape and the skip connection
def test_residual_block_shape():
    block = ResidualBlock(8)
    out = block(torch.randn(2, 8, 8, 8))
    assert out.shape == (2, 8, 8, 8)


#forward returns all three heads with the expected shapes
def test_forward_shapes():
    net = ChessNet(small(lookahead_n=2))
    out = net(torch.randn(4, 102, 8, 8))
    assert out["policy"].shape == (4, policy_size)   # (4, 4672)
    assert out["value"].shape == (4,)
    assert out["lookahead"].shape == (4, 2, policy_size)


#value head output is squashed into [-1, 1]
def test_value_range():
    net = ChessNet(small())
    v = net(torch.randn(8, 102, 8, 8))["value"]
    assert v.min() >= -1.0 and v.max() <= 1.0


#the lookahead horizon in the config drives the lookahead dimension
def test_lookahead_horizon_config():
    net = ChessNet(small(lookahead_n=4))
    out = net(torch.randn(2, 102, 8, 8))
    assert out["lookahead"].shape == (2, 4, policy_size)


#gradients flow to every parameter through all three heads
def test_backward_reaches_all_params():
    net = ChessNet(small(lookahead_n=2))
    out = net(torch.randn(3, 102, 8, 8))
    loss = out["policy"].sum() + out["value"].sum() + out["lookahead"].sum()
    loss.backward()
    missing = [name for name, p in net.named_parameters() if p.grad is None]
    assert missing == [], f"no gradient for: {missing}"
