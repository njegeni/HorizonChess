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

from play import load_model, choose_move


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
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="opening sampling temperature (0 = always best move, >1 = more variety)")
    ap.add_argument("--sample-plies", type=int, default=20,
                    help="sample with temperature for this many opening plies, then play best")
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
            # Declare the options a GUI / lichess-bot commonly sets. We don't act
            # on them (a bare policy has nothing to tune), but they must be
            # declared or the client refuses to send setoption for them.
            send("option name Move Overhead type spin default 10 min 0 max 10000")
            send("option name Threads type spin default 1 min 1 max 512")
            send("option name Hash type spin default 16 min 1 max 65536")
            send("option name Ponder type check default false")
            send("option name SyzygyPath type string default <empty>")
            send("option name UCI_Chess960 type check default false")
            send("option name UCI_Variant type string default chess")
            send("uciok")
        elif cmd.startswith("setoption"):
            pass                             # accept and ignore all options
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
                # sample the opening for variety, then play the best move
                temp = args.temperature if len(board.move_stack) < args.sample_plies else 0.0
                move, _ = choose_move(net, board, temp, args.device)
                send(f"bestmove {move.uci()}")
        elif cmd == "quit":
            break


if __name__ == "__main__":
    main()
