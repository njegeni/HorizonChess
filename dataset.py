"""
Streaming training data from PGN files.

Turns a PGN archive into a stream of training examples for the network:

    (input_tensor, policy_target, value_target)

- input_tensor  : the 102x8x8 board encoding from `encoding.board_to_tensor`,
                  computed at each position *before* the played move.
- policy_target : the index of the move actually played, from
                  `encoding.move_to_index` (an int in [0, POLICY_SIZE)).
- value_target  : the game outcome from the side-to-move's perspective,
                  +1.0 win / 0.0 draw / -1.0 loss.

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
    """Streams (input, policy_target, value_target) tensors from a PGN file.

    Each worker in a multi-worker DataLoader reads the same file independently
    and keeps only every Nth game (N = num_workers), offset by its worker id,
    so the workers together cover every game exactly once.
    """

    def __init__(self, pgn_path: str, skip_unfinished: bool = True):
        super().__init__()
        self.pgn_path = pgn_path
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

                board = game.board()
                for move in game.mainline_moves():
                    tensor = torch.from_numpy(board_to_tensor(board))
                    policy_index = move_to_index(move, board.turn)
                    value = white_value if board.turn == chess.WHITE else -white_value
                    yield (
                        tensor,
                        torch.tensor(policy_index, dtype=torch.long),
                        torch.tensor(value, dtype=torch.float32),
                    )
                    board.push(move)


if __name__ == "__main__":
    # Quick smoke test: pull a handful of examples and show their shapes.
    from torch.utils.data import DataLoader

    dataset = PGNDataset("pgnmentor.pgn")
    loader = DataLoader(dataset, batch_size=8)

    inputs, policies, values = next(iter(loader))
    print("input batch :", inputs.shape, inputs.dtype)
    print("policy batch:", policies.shape, policies.dtype, policies[:8].tolist())
    print("value batch :", values.shape, values.dtype, values[:8].tolist())
