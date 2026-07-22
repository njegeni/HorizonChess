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

import mcts
from play import load_model, choose_move


def send(line):
    print(line, flush=True)          # UCI requires flushing after every reply


def compute_max_time(cmd, board):
    """From a `go` command, return a per-move time budget in seconds (or None to
    fall back to a fixed sim count). Spends a fraction of the remaining clock plus
    most of the increment, capped so it never burns too much at once, with a small
    safety margin so it can't flag."""
    toks = cmd.split()
    p = {}
    for i in range(1, len(toks) - 1):
        if toks[i] in ("wtime", "btime", "winc", "binc", "movetime", "movestogo"):
            try:
                p[toks[i]] = int(toks[i + 1])
            except ValueError:
                pass
    if "movetime" in p:
        return max(p["movetime"] - 100, 30) / 1000.0
    if "wtime" in p or "btime" in p:
        my_time = p.get("wtime" if board.turn == chess.WHITE else "btime", 0)
        my_inc = p.get("winc" if board.turn == chess.WHITE else "binc", 0)
        mtg = max(p.get("movestogo", 25), 1)
        budget = my_time / mtg + my_inc * 0.75      # ms
        budget = min(budget, my_time * 0.4)         # never blow >40% of the clock
        return max(budget - 100, 30) / 1000.0       # 100 ms safety, 30 ms floor
    return None


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
    ap.add_argument("--sims", type=int, default=100,
                    help="fallback MCTS sims when the GUI sends no clock (0 = raw policy). "
                         "with a clock, sims scale to the time budget instead.")
    ap.add_argument("--c-puct", type=float, default=1.5,
                    help="MCTS exploration constant")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="leaves per net eval (1 is fastest for this small net)")
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
                max_time = compute_max_time(cmd, board)   # None if GUI sent no clock
                if max_time is not None or args.sims > 0:
                    # clock present -> time governs (huge sim cap); else fixed sims
                    sims_cap = 100_000 if max_time is not None else args.sims
                    move, _, _ = mcts.best_move(net, board, sims=sims_cap,
                                                c_puct=args.c_puct, temperature=temp,
                                                device=args.device, max_time=max_time,
                                                batch_size=args.batch_size)
                else:
                    move, _ = choose_move(net, board, temp, args.device)
                send(f"bestmove {move.uci()}")
        elif cmd == "quit":
            break


if __name__ == "__main__":
    main()
