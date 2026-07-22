"""
UCI adapter for the trained HorizonChess model.

Speaks just enough of the UCI protocol for a chess GUI or lichess-bot to drive
the net. It plays the policy head's top *legal* move (no search), so time-control
arguments in `go` are ignored -- a bare policy plays instantly.

    python uci.py --ckpt weights/datashuffling/step_00301081.pt
    # or, for lichess-bot which launches the engine with no args:
    HORIZON_CKPT=weights/datashuffling/step_00301081.pt python uci.py
"""

import argparse
import os
import sys

import chess

from play import load_model, rank_moves


def send(line):
    print(line, flush=True)          # UCI requires flushing after every reply


def parse_position(cmd):
    # position [startpos | fen <6-field FEN>] [moves m1 m2 ...]
    tokens = cmd.split()
    if "startpos" in tokens:
        board = chess.Board()
        i = tokens.index("startpos") + 1
    elif "fen" in tokens:
        f = tokens.index("fen") + 1
        board = chess.Board(" ".join(tokens[f:f + 6]))
        i = f + 6
    else:
        board = chess.Board()
        i = len(tokens)
    if i < len(tokens) and tokens[i] == "moves":
        for uci in tokens[i + 1:]:
            board.push_uci(uci)
    return board


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.environ.get("HORIZON_CKPT"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if not args.ckpt:
        sys.exit("no checkpoint: pass --ckpt or set HORIZON_CKPT")

    net = load_model(args.ckpt, args.device)
    board = chess.Board()

    for line in sys.stdin:
        cmd = line.strip()
        if cmd == "uci":
            send("id name HorizonChess")
            send("id author noli")
            send("uciok")
        elif cmd == "isready":
            send("readyok")
        elif cmd == "ucinewgame":
            board = chess.Board()
        elif cmd.startswith("position"):
            board = parse_position(cmd)
        elif cmd.startswith("go"):
            if board.is_game_over():
                send("bestmove 0000")            # null move: game already over
            else:
                ranked, _ = rank_moves(net, board, args.device)
                send(f"bestmove {ranked[0][0].uci()}")
        elif cmd == "quit":
            break


if __name__ == "__main__":
    main()
