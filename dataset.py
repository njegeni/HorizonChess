"""
Streaming training data from PGN files.

Turns a PGN archive into a stream of training examples for the network. Each
example is a dict (so a default-collated batch maps straight onto the model
input and the `loss.compute_loss` targets):

    {
      "input"          : (102, 8, 8) board encoding from `board_to_tensor`,
                         computed at the position *before* the played move.
      "policy"         : () long -- index of the move actually played, from
                         `move_to_index` (in [0, POLICY_SIZE)).
      "value"          : () float -- game outcome from the side-to-move's
                         perspective: +1.0 win / 0.0 draw / -1.0 loss.
      "lookahead"      : (n,) long -- indices of the moves played n plies ahead
                         (plies t+1 .. t+n). Encoded in each future mover's OWN
                         frame, exactly like `policy`, so the lookahead target at
                         (t, k) equals the policy target at position t+k.
      "lookahead_mask" : (n,) bool -- False for horizons that run past the end of
                         the game (target undefined there).
    }

`n` is `lookahead_horizon` and must match the model's `ModelConfig` value.

Odd horizons (t+1, t+3, ...) are the opponent's moves and even horizons
(t+2, t+4, ...) are the side-to-move's own later moves -- the parity the aux
loss weights differently. Because each future move is encoded in its own
mover's frame, even-horizon targets share the current position's orientation
and odd-horizon targets are in the flipped (opponent) orientation; the model's
per-k embedding lets the lookahead head tell them apart.

The dataset is iterable (not indexable): the 2.5 GB PGN is read one game at a
time so the whole archive never has to sit in memory. Under a multi-worker
DataLoader each worker strides over a disjoint set of games, so no example is
produced twice.
"""

import chess
import chess.pgn
import torch
from torch.utils.data import IterableDataset, get_worker_info

from encoding import board_to_tensor, move_to_index

# Game result string -> value from White's perspective.
_RESULT_TO_WHITE_VALUE = {
    "1-0": 1.0,
    "0-1": -1.0,
    "1/2-1/2": 0.0,
}


class PGNDataset(IterableDataset):
    """Streams training-example dicts from a PGN file.

    Each worker in a multi-worker DataLoader reads the same file independently
    and keeps only every Nth game (N = num_workers), offset by its worker id,
    so the workers together cover every game exactly once.
    """

    def __init__(self, pgn_path: str, lookahead_horizon: int = 2,
                 skip_unfinished: bool = True):
        super().__init__()
        self.pgn_path = pgn_path
        self.lookahead_horizon = lookahead_horizon
        self.skip_unfinished = skip_unfinished

    def __iter__(self):
        worker = get_worker_info()
        num_workers = worker.num_workers if worker is not None else 1
        worker_id = worker.id if worker is not None else 0

        with open(self.pgn_path, "r", encoding="utf-8", errors="replace") as pgn_file:
            game_index = 0
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                # Stride: this worker only handles its share of games.
                if game_index % num_workers != worker_id:
                    game_index += 1
                    continue
                game_index += 1

                white_value = _RESULT_TO_WHITE_VALUE.get(game.headers.get("Result"))
                if white_value is None:
                    if self.skip_unfinished:
                        continue
                    white_value = 0.0

                yield from self._game_examples(game, white_value)

    def _game_examples(self, game: chess.pgn.Game, white_value: float):
        """Yield one example dict per position in a single game."""
        moves = list(game.mainline_moves())
        if not moves:
            return
        n = self.lookahead_horizon

        # First pass (cheap, no tensors): encode every played move in its own
        # mover's frame. policy_indices[j] doubles as the lookahead target for
        # any earlier position that looks j plies ahead.
        policy_indices = []
        board = game.board()
        for move in moves:
            policy_indices.append(move_to_index(move, board.turn))
            board.push(move)

        # Second pass: build the input tensor (needs the live board so its move
        # stack feeds the history planes) and attach the lookahead window.
        num_moves = len(moves)
        board = game.board()
        for t, move in enumerate(moves):
            tensor = torch.from_numpy(board_to_tensor(board))
            value = white_value if board.turn == chess.WHITE else -white_value

            lookahead = torch.zeros(n, dtype=torch.long)
            lookahead_mask = torch.zeros(n, dtype=torch.bool)
            for k in range(1, n + 1):
                j = t + k
                if j < num_moves:
                    lookahead[k - 1] = policy_indices[j]
                    lookahead_mask[k - 1] = True

            yield {
                "input": tensor,
                "policy": torch.tensor(policy_indices[t], dtype=torch.long),
                "value": torch.tensor(value, dtype=torch.float32),
                "lookahead": lookahead,
                "lookahead_mask": lookahead_mask,
            }
            board.push(move)


if __name__ == "__main__":
    # Quick smoke test: pull a handful of examples and show their shapes.
    from torch.utils.data import DataLoader

    dataset = PGNDataset("pgnmentor.pgn", lookahead_horizon=2)
    loader = DataLoader(dataset, batch_size=8)

    batch = next(iter(loader))
    for key, tensor in batch.items():
        print(f"{key:15s}: {tuple(tensor.shape)} {tensor.dtype}")
    print("policy sample :", batch["policy"][:8].tolist())
    print("value sample  :", batch["value"][:8].tolist())
    print("lookahead[0]  :", batch["lookahead"][0].tolist())
    print("mask[0]       :", batch["lookahead_mask"][0].tolist())
