"""
Shuffle a PGN at the game level, once, so training sees a stationary
distribution.

PGN Mentor archives group games by player/opening and concatenate them, so the
file is ordered, not random. The dataset's per-worker shuffle buffer only mixes
a few hundred nearby games, which isn't enough to undo that global ordering --
so the training distribution drifts as it reads through the file and validation
(drawn from the front) diverges. Shuffling the games once up front fixes this at
the source.

Memory-light on purpose (the file is ~2.5 GB): one pass records the byte offset
of every game (each starts with a line beginning "[Event "), the offset list is
shuffled, then each game's byte range is copied to the output in the new order.
Only the offsets live in memory, never the whole file.

    python shuffle_pgn.py pgnmentor.pgn pgnmentor_shuffled.pgn --seed 0
"""

import argparse
import random

GAME_START = b"[Event "


def find_game_bounds(path):
    """Return a list of (start, stop) byte ranges, one per game."""
    starts = []
    with open(path, "rb") as f:
        pos = 0
        for line in f:
            if line.startswith(GAME_START):
                starts.append(pos)
            pos += len(line)
        end = pos
    # each game spans from its [Event up to the next game's [Event (or EOF)
    return [(starts[i], starts[i + 1] if i + 1 < len(starts) else end)
            for i in range(len(starts))]


def main():
    ap = argparse.ArgumentParser(description="Game-level shuffle of a PGN file.")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bounds = find_game_bounds(args.input)
    if not bounds:
        raise SystemExit("no games found (no lines starting with '[Event ')")
    print(f"found {len(bounds):,} games")

    random.Random(args.seed).shuffle(bounds)

    written = 0
    with open(args.input, "rb") as src, open(args.output, "wb") as dst:
        for start, stop in bounds:
            src.seek(start)
            dst.write(src.read(stop - start))
            written += 1
            if written % 100_000 == 0:
                print(f"  wrote {written:,}/{len(bounds):,} games")
    print(f"done -> {args.output} ({written:,} games)")


if __name__ == "__main__":
    main()
