"""
PUCT Monte-Carlo Tree Search over the policy+value net (AlphaZero-style).

The net alone plays its favorite move with no lookahead, so it blunders
tactically. MCTS uses the same net to *search*: the policy suggests which moves
to explore, the value scores leaf positions, and after N simulations the move
that actually holds up under search is played.

Each node's stats are stored from the perspective of the side to move at that
node (negamax): a child's Q is from the opponent's view, so selection uses
-child.Q. Leaf values come from the value head (side-to-move view); terminal
positions score -1 for a side-to-move that is checkmated, 0 for a draw.

Single-leaf evaluation for clarity; the obvious speedup later is to batch leaf
net evals. Cost scales with --sims (each sim = one net forward pass).
"""

import math

import chess
import torch
import torch.nn.functional as F

from encoding import board_to_tensor, encode_move


class Node:
    __slots__ = ("prior", "children", "N", "W", "is_expanded", "is_terminal")

    def __init__(self, prior):
        self.prior = prior
        self.children = {}          # move -> Node
        self.N = 0                  # visit count
        self.W = 0.0                # total value (this node's side-to-move view)
        self.is_expanded = False
        self.is_terminal = False

    @property
    def Q(self):
        return self.W / self.N if self.N else 0.0


@torch.no_grad()
def evaluate(net, board, device):
    """Return (priors: {move: prob} over legal moves, value in [-1,1])."""
    x = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(device)
    out = net(x)
    logits = out["policy"][0]
    value = out["value"].item()
    legal = list(board.legal_moves)
    idx = torch.tensor([encode_move(m, board) for m in legal], device=device)
    probs = F.softmax(logits[idx], dim=0).tolist()
    return dict(zip(legal, probs)), value


def _select_child(node, c_puct):
    total = math.sqrt(node.N)
    best_score, best = -1e30, None
    for move, child in node.children.items():
        # child.Q is from the child's side-to-move view (the opponent), so the
        # value to *this* node of picking it is -child.Q.
        u = c_puct * child.prior * total / (1 + child.N)
        score = -child.Q + u
        if score > best_score:
            best_score, best = score, (move, child)
    return best


def search(net, board, sims=200, c_puct=1.5, device="cpu"):
    """Run `sims` simulations from `board`; return the root Node."""
    root = Node(prior=1.0)
    priors, _ = evaluate(net, board, device)
    for mv, p in priors.items():
        root.children[mv] = Node(p)
    root.is_expanded = True

    for _ in range(sims):
        node = root
        scratch = board.copy()
        path = [node]

        # SELECT: descend by PUCT until we reach a leaf (unexpanded) node
        while node.is_expanded and not node.is_terminal:
            move, node = _select_child(node, c_puct)
            scratch.push(move)
            path.append(node)

        # EVALUATE: terminal result, or expand the leaf with the net
        if scratch.is_game_over():
            node.is_terminal = True
            value = -1.0 if scratch.is_checkmate() else 0.0
        else:
            priors, value = evaluate(net, scratch, device)
            for mv, p in priors.items():
                node.children[mv] = Node(p)
            node.is_expanded = True

        # BACKUP: flip sign each ply (value is from each node's own view)
        for n in reversed(path):
            n.N += 1
            n.W += value
            value = -value

    return root


def best_move(net, board, sims=200, c_puct=1.5, temperature=0.0, device="cpu"):
    """Search, then pick a move by visit count (the MCTS-improved policy).
    temperature>0 samples proportional to visits^(1/T) for opening variety."""
    root = search(net, board, sims, c_puct, device)
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
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    net = load_model(a.ckpt, a.device)
    board = chess.Board(a.fen)
    root = search(net, board, a.sims, device=a.device)

    ranked = sorted(root.children.items(), key=lambda kv: kv[1].N, reverse=True)
    print(f"root value: {root.Q:+.3f}   ({a.sims} sims)")
    print("top moves by visits:")
    for move, node in ranked[:6]:
        print(f"  {board.san(move):6s}  N={node.N:4d}  Q={node.Q:+.3f}  P={node.prior:.3f}")
