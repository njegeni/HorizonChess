import io
import chess, chess.pgn
import torch
from dataset import PGNDataSet
from encoding import encode_move


#a small in-memory PGN: game 0 is short, game 2 is illegal (skipped)
GAMES = """[Event "g0"]
[Result "1-0"]

1. e4 e5 1-0

[Event "g1"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0

[Event "g2-illegal"]
[Result "0-1"]

1. e4 e5 2. Nf3 Qxe4 0-1

[Event "g3"]
[Result "1/2-1/2"]

1. d4 d5 2. c4 e6 1/2-1/2
"""


def write_pgn(tmp_path):
    p = tmp_path / "games.pgn"
    p.write_text(GAMES)
    return str(p)


def one_game(pgn_text):
    return chess.pgn.read_game(io.StringIO(pgn_text))


#each example has the right keys, shapes and dtypes
def test_example_structure(tmp_path):
    ds = PGNDataSet(write_pgn(tmp_path), lookahead_horizon=2)
    ex = next(iter(ds))
    assert ex["input"].shape == (102, 8, 8) and ex["input"].dtype == torch.float32
    assert ex["policy"].dtype == torch.long and ex["policy"].shape == ()
    assert ex["value"].dtype == torch.float32
    assert ex["lookahead"].shape == (2,) and ex["lookahead"].dtype == torch.long
    assert ex["lookahead_mask"].shape == (2,) and ex["lookahead_mask"].dtype == torch.bool


#value is from the side-to-move's view: a white-won game alternates +1, -1
def test_value_perspective():
    ds = PGNDataSet("unused", lookahead_horizon=2)
    game = one_game("""[Event "x"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0
""")
    examples = list(ds._game_examples(game, white_value=1.0))
    values = [e["value"].item() for e in examples]
    assert values == [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]   # 6 plies, white to move first


#the lookahead target at (t, k) equals the policy target k plies later; the
#mask is False once the game runs out of moves
def test_lookahead_invariant():
    n = 3
    ds = PGNDataSet("unused", lookahead_horizon=n)
    game = one_game("""[Event "x"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0
""")
    examples = list(ds._game_examples(game, white_value=1.0))
    policies = [e["policy"].item() for e in examples]
    num = len(examples)
    for t, e in enumerate(examples):
        for k in range(1, n + 1):
            j = t + k
            valid = j < num
            assert bool(e["lookahead_mask"][k - 1]) == valid
            if valid:
                assert e["lookahead"][k - 1].item() == policies[j]


#corrupt games are skipped; train and val splits partition the valid games
def test_split_and_corrupt(tmp_path):
    path = write_pgn(tmp_path)
    train = list(PGNDataSet(path, split="train", val_every=50))
    val = list(PGNDataSet(path, split="val", val_every=50))
    # g0 (idx 0) is the only val game (2 plies); g1 (6) + g3 (4) are train;
    # g2 is illegal and skipped from both.
    assert len(val) == 2
    assert len(train) == 10
    # together they cover every valid position exactly once
    assert len(train) + len(val) == 12
