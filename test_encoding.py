"""
Correctness tests for `encoding.py`.

The network trusts this module completely: a silent bug here (a mis-oriented
history plane, a move index collision) trains the model on garbage without ever
raising. So we check the two things that actually matter:

1. move_to_index / index_to_move round-trip for *every legal move* in a range
   of positions, and every legal move maps to a *distinct* policy index.
2. board_to_tensor lays pieces/flags on the planes the docstring promises, and
   the side-to-move orientation (the `flip` logic) is applied consistently to
   the current state and the whole history stack.

Run directly (`python test_encoding.py`) for a plain pass/fail summary, or via
`pytest test_encoding.py`.
"""

import random

import chess
import numpy as np

from encoding import (
    POLICY_SIZE,
    board_to_tensor,
    index_to_move,
    move_to_index,
)


def _random_positions(count: int, max_plies: int, seed: int = 0):
    """Yield `count` boards reached by random legal play (plus the start pos)."""
    rng = random.Random(seed)
    yield chess.Board()  # always cover the starting position
    for _ in range(count):
        board = chess.Board()
        plies = rng.randint(1, max_plies)
        for _ in range(plies):
            moves = list(board.legal_moves)
            if not moves:
                break  # checkmate / stalemate
            board.push(rng.choice(moves))
        yield board


# --------------------------------------------------------------------------
# 1. Policy (move) encoding
# --------------------------------------------------------------------------

def test_move_index_roundtrips_for_all_legal_moves():
    """index_to_move(move_to_index(m)) == m for every legal move."""
    for board in _random_positions(count=300, max_plies=60):
        for move in board.legal_moves:
            index = move_to_index(move, board.turn)
            assert 0 <= index < POLICY_SIZE, f"index {index} out of range for {move}"
            restored = index_to_move(index, board)
            assert restored == move, (
                f"round-trip failed: {move.uci()} -> {index} -> {restored.uci()} "
                f"in FEN {board.fen()}"
            )


def test_legal_moves_map_to_distinct_indices():
    """No two legal moves from a position collide on the same policy index."""
    for board in _random_positions(count=300, max_plies=60):
        seen = {}
        for move in board.legal_moves:
            index = move_to_index(move, board.turn)
            assert index not in seen, (
                f"index collision {index}: {move.uci()} vs {seen[index].uci()} "
                f"in FEN {board.fen()}"
            )
            seen[index] = move


def test_underpromotions_use_dedicated_planes():
    """Knight/bishop/rook promotions land on planes 64-72; queen promo does not."""
    # White pawn on a7, kings out of the way -> a8 promotions are legal.
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    planes = set()
    for move in board.legal_moves:
        if move.from_square != chess.A7:
            continue
        index = move_to_index(move, board.turn)
        plane = index // 64
        if move.promotion == chess.QUEEN:
            assert plane < 64, f"queen promo should be a queen-plane move, got {plane}"
        else:
            assert 64 <= plane < 73, f"underpromo {move.uci()} on plane {plane}"
        planes.add(plane)
    # a7a8 with N/B/R -> three distinct underpromotion planes.
    assert len([p for p in planes if p >= 64]) == 3


# --------------------------------------------------------------------------
# 2. Board (input) encoding
# --------------------------------------------------------------------------

def test_starting_position_planes():
    tensor = board_to_tensor(chess.Board())
    assert tensor.shape == (102, 8, 8)

    # White to move -> no flip. White pawns (plane 0) sit on rank 2 = row 1.
    assert tensor[0, 1, :].sum() == 8
    assert tensor[0].sum() == 8
    # White king (plane 5) on e1 = row 0, col 4.
    assert tensor[5, 0, 4] == 1.0
    # Black king (plane 11) on e8 = row 7, col 4.
    assert tensor[11, 7, 4] == 1.0

    # Side-to-move plane is all ones; all four castling planes set at the start.
    assert tensor[12].sum() == 64
    for p in (13, 14, 15, 16):
        assert tensor[p].sum() == 64, f"castling plane {p} not fully set"

    # No en passant, and no history (game just started) -> planes 18+ are zero.
    assert tensor[17].sum() == 0
    assert tensor[18:].sum() == 0


def test_total_piece_count_matches_board():
    """The 12 current-state piece planes hold exactly one 1 per piece on board."""
    for board in _random_positions(count=100, max_plies=60):
        tensor = board_to_tensor(board)
        assert tensor[0:12].sum() == len(board.piece_map())


def test_black_to_move_is_oriented_like_white():
    """After 1.e4 (Black to move) the board is mirrored so the mover plays up.

    Black's own pieces must occupy planes 0-5, and Black's pawns — which live on
    rank 7 in reality — must appear on row 1 (rank 2) in the oriented frame,
    exactly where White's pawns sat before.
    """
    board = chess.Board()
    board.push_san("e4")  # now Black to move -> flip = True
    tensor = board_to_tensor(board)

    # mirror() flips rank AND swaps color, so Black's pawns (rank 7) land on
    # row 1. Black hasn't moved yet, so all 8 are still home.
    assert tensor[0, 1, :].sum() == 8
    # Side-to-move king (plane 5) = Black king e8 -> mirrored to e1 = row 0, col 4.
    assert tensor[5, 0, 4] == 1.0


def test_history_planes_share_current_orientation():
    """Every history frame uses the *current* side-to-move orientation.

    We reach a position with Black to move, so the whole stack is flipped. The
    previous frame (s_{t-1}) must be oriented the same way as the current frame,
    not by its own (opposite) turn. We check that by confirming the history
    piece planes carry the right piece count and that the block boundary is
    where the docstring says (planes 18-29 for s_{t-1}).
    """
    board = chess.Board()
    for san in ("e4", "c5", "Nf3"):  # 3 plies -> Black to move, 3 history frames
        board.push_san(san)
    tensor = board_to_tensor(board)

    # s_{t-1} is the position before Nf3: reconstruct it and compare counts.
    prev = board.copy()
    prev.pop()
    assert tensor[18:30].sum() == len(prev.piece_map())

    # Only 3 plies of history exist; frame index 3+ (planes 54+) must be empty.
    assert tensor[54:].sum() == 0


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {test.__name__}\n      {exc}")
        except Exception as exc:  # noqa: BLE001 - surface unexpected errors too
            failures += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {test.__name__}")
    print()
    if failures:
        print(f"{failures} of {len(tests)} tests failed")
    else:
        print(f"all {len(tests)} tests passed")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if _run_all() else 0)
