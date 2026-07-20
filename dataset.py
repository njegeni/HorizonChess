import logging
import chess, chess.pgn
import torch
from torch.utils.data import IterableDataset, get_worker_info
from encoding import board_to_tensor, encode_move

#logging corrupt games (typos, illegal moves)
logging.getLogger("chess.pgn").setLevel(logging.CRITICAL)

#translates the games result string into the value target
result_to_white_value = {"1-0": 1.0, #white wins
                          "0-1": -1.0,  #black wins
                          "1/2-1/2": 0.0 #draw
                        }

class PGNDataSet(IterableDataset):
    """
    Streams training examples from a PGN file. Each example is a dict:
        input          : (102, 8, 8) board tensor from board_to_tensor
        policy         : () long  -- index of the move played (encode_move)
        value          : () float -- game result from the side-to-move's view
        lookahead      : (n,) long -- indices of the moves played n plies ahead
        lookahead_mask : (n,) bool -- False where the game ended before that ply

    split='val' holds out every val_every-th game (by global index); split='train'
    keeps the rest, so the two are disjoint. Under a multi-worker DataLoader each
    worker strides over its own share of games so no example is produced twice.
    """

    def __init__(self, pgn_path, lookahead_horizon=2, split="train", val_every=50):
        super().__init__()
        self.pgn_path = pgn_path
        self.lookahead_horizon = lookahead_horizon
        self.split = split
        self.val_every = val_every

    def __iter__(self):
        worker = get_worker_info()
        num_workers = worker.num_workers if worker is not None else 1
        worker_id = worker.id if worker is not None else 0

        with open(self.pgn_path, encoding="utf-8", errors="replace") as f:
            game_index = 0
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                idx = game_index
                game_index += 1

                # each worker only handles its share of games
                if idx % num_workers != worker_id:
                    continue
                # train/val holdout: keep only the games for this split
                is_val = idx % self.val_every == 0
                if (self.split == "val") != is_val:
                    continue
                # skip corrupted games (illegal move, truncated mainline)
                if game.errors:
                    continue

                white_value = result_to_white_value.get(game.headers.get("Result"))
                if white_value is None:
                    continue

                yield from self._game_examples(game, white_value)

    def _game_examples(self, game, white_value):
        moves = list(game.mainline_moves())
        if not moves:
            return
        n = self.lookahead_horizon

        # first pass (cheap, no tensors): encode every played move in its own
        # mover's frame. policy_indices[j] also serves as the lookahead target
        # for any earlier position that looks j plies ahead.
        policy_indices = []
        board = game.board()
        for move in moves:
            policy_indices.append(encode_move(move, board))
            board.push(move)

        # second pass: build the input tensor (needs the live board so its move
        # stack feeds the history planes) and attach the lookahead window.
        num_moves = len(moves)
        board = game.board()
        for t, move in enumerate(moves):
            value = white_value if board.turn == chess.WHITE else -white_value

            lookahead = torch.zeros(n, dtype=torch.long)
            lookahead_mask = torch.zeros(n, dtype=torch.bool)
            for k in range(1, n + 1):
                j = t + k
                if j < num_moves:
                    lookahead[k - 1] = policy_indices[j]
                    lookahead_mask[k - 1] = True

            yield {
                "input": torch.from_numpy(board_to_tensor(board)),
                "policy": torch.tensor(policy_indices[t], dtype=torch.long),
                "value": torch.tensor(value, dtype=torch.float32),
                "lookahead": lookahead,
                "lookahead_mask": lookahead_mask,
            }
            board.push(move)  # push AFTER yielding (encode the pre-move board)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    loader = DataLoader(PGNDataSet("pgnmentor.pgn", lookahead_horizon=2), batch_size=8)
    batch = next(iter(loader))
    for k, v in batch.items():
        print(k, tuple(v.shape), v.dtype)
    print("policy sample:", batch["policy"][:8].tolist())
    print("value sample :", batch["value"][:8].tolist())
    print("lookahead[0] :", batch["lookahead"][0].tolist())
    print("mask[0]      :", batch["lookahead_mask"][0].tolist())
