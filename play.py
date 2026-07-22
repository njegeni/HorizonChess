"""
Inference / play with a trained HorizonChess model.

Loads a checkpoint and, for a given position, ranks every *legal* move with the
policy head (illegal moves can't be chosen because we only score legal ones) and
reports the value head's evaluation of the position (from the side-to-move's
view, in [-1, 1]).

    python play.py --ckpt step_00301081.pt                       # self-play demo
    python play.py --ckpt step_00301081.pt --fen "<FEN>"         # top moves for a position
    python play.py --ckpt step_00301081.pt --play                # you (white) vs the model
"""

import argparse

import chess
import torch
import torch.nn.functional as F

from encoding import board_to_tensor, encode_move
from model import ChessNet, ModelConfig


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    cfg = ModelConfig(**ckpt["model_config"])
    net = ChessNet(cfg).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net


@torch.no_grad()
def rank_moves(net, board, device="cpu"):
    """Return (ranked_moves, value): a list of (move, probability) sorted best
    first over the LEGAL moves, plus the value-head score for the position."""
    x = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(device)
    out = net(x)
    logits = out["policy"][0]                     # (4672,)
    value = out["value"].item()

    legal = list(board.legal_moves)
    idx = torch.tensor([encode_move(m, board) for m in legal], device=device)
    probs = F.softmax(logits[idx], dim=0)         # softmax over legal moves only
    order = torch.argsort(probs, descending=True)
    ranked = [(legal[i], probs[i].item()) for i in order.tolist()]
    return ranked, value


@torch.no_grad()
def choose_move(net, board, temperature=0.0, device="cpu"):
    """Pick a move. temperature<=0 -> always the top move (deterministic).
    temperature>0 -> sample from the policy, sharpened/flattened by temperature
    (t=1 samples proportional to the policy; t<1 favors the best move; t>1 is
    more random). Used to vary the opening so it doesn't repeat every game."""
    ranked, value = rank_moves(net, board, device)
    if temperature <= 0 or len(ranked) == 1:
        return ranked[0][0], value
    probs = torch.tensor([p for _, p in ranked], dtype=torch.float)
    weights = probs ** (1.0 / temperature)          # == softmax(logits / temperature)
    idx = torch.multinomial(weights, 1).item()
    return ranked[idx][0], value


def show_top(net, board, k, device):
    ranked, value = rank_moves(net, board, device)
    print(f"\nposition value (side to move): {value:+.3f}")
    print(f"top {min(k, len(ranked))} moves:")
    for move, p in ranked[:k]:
        print(f"  {board.san(move):6s}  p={p:.3f}")


def self_play(net, max_plies, device):
    board = chess.Board()
    print("self-play (model picks its own best move each side):\n")
    while not board.is_game_over() and len(board.move_stack) < max_plies:
        ranked, value = rank_moves(net, board, device)
        move, p = ranked[0]
        tag = f"{board.fullmove_number}." if board.turn == chess.WHITE else f"{board.fullmove_number}..."
        print(f"{tag:6s} {board.san(move):6s}  (p={p:.2f}, value={value:+.2f})")
        board.push(move)
    print("\nresult:", board.result())
    print(board)


def play_human(net, device):
    board = chess.Board()
    print("you are White. enter moves in SAN (e.g. e4, Nf3) or 'quit'.\n")
    while not board.is_game_over():
        print(board, "\n")
        if board.turn == chess.WHITE:
            uci = input("your move: ").strip()
            if uci in ("quit", "q"):
                return
            try:
                board.push_san(uci)
            except ValueError:
                print("illegal / unparseable move, try again")
                continue
        else:
            ranked, value = rank_moves(net, board, device)
            move, p = ranked[0]
            print(f"model plays {board.san(move)} (p={p:.2f}, value={value:+.2f})\n")
            board.push(move)
    print("\nresult:", board.result())


def main():
    ap = argparse.ArgumentParser(description="Play / inference with a trained model.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fen", default=None, help="score the top moves for this position")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--play", action="store_true", help="play a game vs the model (you are White)")
    ap.add_argument("--max-plies", type=int, default=120, help="cap for self-play demo")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    net = load_model(args.ckpt, args.device)

    if args.fen:
        show_top(net, chess.Board(args.fen), args.topk, args.device)
    elif args.play:
        play_human(net, args.device)
    else:
        self_play(net, args.max_plies, args.device)


if __name__ == "__main__":
    main()
