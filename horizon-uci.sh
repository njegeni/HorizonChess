#!/usr/bin/env bash
# Launcher so lichess-bot (or any UCI GUI) can run the model as an engine.
# Uses the project's venv python explicitly so torch/chess are always available,
# regardless of the environment lichess-bot itself runs in.
cd /Users/noli/school/Boston/Timemk/HorizonChess
exec ./.venv/bin/python uci.py --ckpt weights/datashuffling/step_00301081.pt \
  --device mps
