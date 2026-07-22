"""
PUCT Monte-Carlo Tree Search over the policy+value net (AlphaZero-style).

The net alone plays its favorite move with no lookahead, so it blunders
tactically. MCTS uses the same net to *search*: the policy suggests which moves
to explore, the value scores leaf positions, and after N simulations the move
that actually holds up under search is played.

Stats are stored per node from the perspective of the side to move at that node
(negamax): a child's Q is the opponent's view, so selection uses -child.Q. Leaf
values come from the value head (side-to-move view); a checkmated side-to-move
scores -1, a draw 0.

Two speed features:
- BATCHED EVAL: leaves are collected `batch_size` at a time and run through the
  net in ONE forward pass (the expensive part), using virtual loss so the
  parallel selections in a batch don't all collapse onto the same leaf.
- TIME BUDGET: `max_time` seconds caps the search, so it thinks as deep as the
  clock allows and never flags.
"""

import math
import time

import chess
import torch
import torch.nn.functional as F

from encoding import board_to_tensor, encode_move

VIRTUAL_LOSS = 1.0


class Node:
    __slots__ = ("prior", "children", "N", "W", "is_expanded", "is_terminal")

    def __init__(self, prior):
        self.prior = prior
        self.children = {}
        self.N = 0
        self.W = 0.0
        self.is_expanded = False
        self.is_terminal = False

    @property
    def Q(self):
        return self.W / self.N if self.N else 0.0


def _priors_from_logits(logits_row, board, device):
    legal = list(board.legal_moves)
    idx = torch.tensor([encode_move(m, board) for m in legal], device=device)
    probs = F.softmax(logits_row[idx], dim=0).tolist()
    return dict(zip(legal, probs))


@torch.no_grad()
def evaluate_batch(net, boards, device):
    """One forward pass over many boards. Returns (list of prior dicts, list of values)."""
    x = torch.stack([torch.from_numpy(board_to_tensor(b)) for b in boards]).to(device)
    out = net(x)
    logits, values = out["policy"], out["value"].tolist()
    priors = [_priors_from_logits(logits[i], b, device) for i, b in enumerate(boards)]
    return priors, values


def _select_child(node, c_puct):
    total = math.sqrt(node.N)
    best_score, best = -1e30, None
    for move, child in node.children.items():
        u = c_puct * child.prior * total / (1 + child.N)
        score = -child.Q + u            # child.Q is the opponent's view
        if score > best_score:
            best_score, best = score, (move, child)
    return best


def search(net, board, sims=200, c_puct=1.5, device="cpu",
           max_time=None, batch_size=16):
    root = Node(prior=1.0)
    root_priors, _ = evaluate_batch(net, [board], device)
    for mv, p in root_priors[0].items():
        root.children[mv] = Node(p)
    root.is_expanded = True

    start = time.monotonic()
    done = 0
    while done < sims:
        if max_time is not None and time.monotonic() - start >= max_time:
            break
        n_batch = min(batch_size, sims - done)

        # --- collect a batch of leaves, applying virtual loss along each path ---
        paths = []                      # (path, leaf, scratch_board)
        for _ in range(n_batch):
            node, scratch, path = root, board.copy(), [root]
            while node.is_expanded and not node.is_terminal:
                move, node = _select_child(node, c_puct)
                scratch.push(move)
                path.append(node)
            for nd in path:             # virtual loss: makes this line look bad
                nd.N += 1               # to the other selections in this batch
                nd.W += VIRTUAL_LOSS
            paths.append((path, node, scratch))

        # --- resolve terminals, batch-evaluate the rest in one net call ---
        values = [None] * len(paths)
        eval_items, eval_boards = [], []
        for i, (path, leaf, scratch) in enumerate(paths):
            if leaf.is_terminal or scratch.is_game_over():
                leaf.is_terminal = True
                values[i] = -1.0 if scratch.is_checkmate() else 0.0
            else:
                eval_items.append((i, leaf))
                eval_boards.append(scratch)
        if eval_boards:
            priors, vals = evaluate_batch(net, eval_boards, device)
            for (i, leaf), pr, v in zip(eval_items, priors, vals):
                if not leaf.is_expanded:            # expand once (a node may recur)
                    for mv, p in pr.items():
                        leaf.children[mv] = Node(p)
                    leaf.is_expanded = True
                values[i] = v

        # --- backup: remove virtual loss and add the real (alternating) value ---
        for i, (path, leaf, scratch) in enumerate(paths):
            v = values[i]
            for nd in reversed(path):
                nd.W += v - VIRTUAL_LOSS            # N already counted at collection
                v = -v
        done += n_batch

    return root


def best_move(net, board, sims=200, c_puct=1.5, temperature=0.0, device="cpu",
              max_time=None, batch_size=16):
    """Search, then pick a move by visit count (temperature>0 samples for variety)."""
    root = search(net, board, sims, c_puct, device, max_time, batch_size)
    moves = list(root.children)
    visits = torch.tensor([root.children[m].N for m in moves], dtype=torch.float)
    if temperature <= 0:
        move = moves[int(visits.argmax())]
    else:
        move = moves[int(torch.multinomial(visits ** (1.0 / temperature), 1))]
    return move, root.Q, root


if __name__ == "__main__":
    import argparse
    from play import load_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fen", default=chess.STARTING_FEN)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    net = load_model(a.ckpt, a.device)
    board = chess.Board(a.fen)
    t0 = time.monotonic()
    root = search(net, board, a.sims, device=a.device, batch_size=a.batch_size)
    dt = time.monotonic() - t0

    ranked = sorted(root.children.items(), key=lambda kv: kv[1].N, reverse=True)
    print(f"root value: {root.Q:+.3f}   ({a.sims} sims, batch {a.batch_size}, {dt:.2f}s)")
    for move, node in ranked[:6]:
        print(f"  {board.san(move):6s}  N={node.N:4d}  Q={node.Q:+.3f}  P={node.prior:.3f}")
