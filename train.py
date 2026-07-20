"""
Supervised training loop for the chess network.

Streams positions from a PGN (dataset.PGNDataSet), trains the three-head
model.ChessNet against loss.compute_loss, logs the per-head loss breakdown plus
policy top-1 accuracy, checkpoints periodically, and evaluates on a held-out
validation set.

    python train.py --max-steps 100000 --batch-size 256 --num-workers 4
    python train.py --resume checkpoints/step_00010000.pt
"""

import argparse
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from itertools import islice

import torch
from torch.utils.data import DataLoader

from dataset import PGNDataSet
from loss import compute_loss
from model import ChessNet, ModelConfig


@dataclass
class TrainConfig:
    pgn_path: str = "pgnmentor.pgn"
    batch_size: int = 256
    lr: float = 1e-3
    min_lr: float = 1e-5           # cosine decay floor
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    grad_clip: float = 4.0
    num_workers: int = 4
    max_steps: int = 100_000
    epochs: int = 1
    shuffle_buffer: int = 8192     # decorrelate batches (per worker)

    # loss weights (forwarded to compute_loss)
    value_weight: float = 1.0
    gamma: float = 0.85
    w_opp: float = 0.15
    w_self: float = 0.15

    log_interval: int = 100
    ckpt_interval: int = 10_000
    ckpt_dir: str = "checkpoints"

    # validation
    do_val: bool = True
    val_interval: int = 5000
    val_batches: int = 20
    val_every: int = 50            # 1 in N games held out for validation


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_at(step, cfg):
    # linear warmup, then cosine decay from lr down to min_lr over the run
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(max(progress, 0.0), 1.0)          # clamp for steps past max
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * cosine


class Meter:
    """Running mean of the loss parts + accuracy since the last log."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.sums = {}
        self.count = 0

    def update(self, values):
        for k, v in values.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v)
        self.count += 1

    def averages(self):
        return {k: v / max(self.count, 1) for k, v in self.sums.items()}


def to_targets(batch, device):
    return {
        "policy": batch["policy"].to(device, non_blocking=True),
        "value": batch["value"].to(device, non_blocking=True),
        "lookahead": batch["lookahead"].to(device, non_blocking=True),
        "lookahead_mask": batch["lookahead_mask"].to(device, non_blocking=True),
    }


def build_val_cache(cfg, model_cfg):
    """Materialize a FIXED validation set into memory once.

    Val games are sparse (1 in val_every), so streaming them fresh each eval
    would re-parse a big chunk of the PGN every time. Single process on purpose:
    streaming is fast enough that spawning workers for a one-time scan is not
    worth its re-import cost."""
    ds = PGNDataSet(cfg.pgn_path, lookahead_horizon=model_cfg.lookahead_n,
                    split="val", val_every=cfg.val_every, shuffle_buffer=0)
    loader = DataLoader(ds, batch_size=cfg.batch_size, num_workers=0, drop_last=True)
    print(f"building validation set ({cfg.val_batches} batches x {cfg.batch_size}) "
          f"-- one-time scan of the PGN for held-out games...")
    cache = list(islice(loader, cfg.val_batches))
    print(f"  cached {len(cache)} validation batches ({len(cache) * cfg.batch_size} positions)")
    return cache


@torch.no_grad()
def evaluate(model, val_cache, device, loss_kwargs):
    model.eval()
    meter = Meter()
    for batch in val_cache:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = to_targets(batch, device)
        outputs = model(inputs)
        _, parts = compute_loss(outputs, targets, **loss_kwargs)
        acc = (outputs["policy"].argmax(1) == targets["policy"]).float().mean()
        meter.update({**{k: v.item() for k, v in parts.items()}, "acc": acc.item()})
    model.train()
    return meter.averages()


def save_checkpoint(path, step, model, optimizer, model_cfg, train_cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
    }, path)


def train(cfg, model_cfg, resume=None):
    # line-buffer stdout so progress prints show immediately even when piped
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    device = select_device()
    use_amp = device.type == "cuda"
    print(f"device: {device} | AMP: {use_amp}")

    # dataset and model must agree on the aux-head horizon
    dataset = PGNDataSet(cfg.pgn_path, lookahead_horizon=model_cfg.lookahead_n,
                         split="train", val_every=cfg.val_every,
                         shuffle_buffer=cfg.shuffle_buffer)
    assert dataset.lookahead_horizon == model_cfg.lookahead_n

    loss_kwargs = dict(value_weight=cfg.value_weight, gamma=cfg.gamma,
                       w_opp=cfg.w_opp, w_self=cfg.w_self)

    loader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                        pin_memory=(device.type == "cuda"),
                        persistent_workers=cfg.num_workers > 0, drop_last=True)

    val_cache = build_val_cache(cfg, model_cfg) if cfg.do_val else None

    model = ChessNet(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    step = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        print(f"resumed from {resume} at step {step}")

    model.train()
    meter = Meter()
    t0 = time.time()

    for _epoch in range(cfg.epochs):
        for batch in loader:
            if step >= cfg.max_steps:
                break

            inputs = batch["input"].to(device, non_blocking=True)
            targets = to_targets(batch, device)

            for g in optimizer.param_groups:
                g["lr"] = lr_at(step, cfg)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
                total, parts = compute_loss(outputs, targets, **loss_kwargs)

            scaler.scale(total).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                acc = (outputs["policy"].argmax(1) == targets["policy"]).float().mean()
            meter.update({**{k: v.item() for k, v in parts.items()}, "acc": acc.item()})
            step += 1

            if step % cfg.log_interval == 0:
                avg = meter.averages()
                rate = cfg.log_interval * cfg.batch_size / (time.time() - t0)
                print(f"step {step:>7} | lr {lr_at(step, cfg):.2e} | "
                      f"total {avg['total']:.3f} | policy {avg['policy']:.3f} "
                      f"(acc {avg['acc']:.3f}) | value {avg['value']:.3f} | "
                      f"aux {avg['aux']:.4f} | {rate:.0f} pos/s")
                meter.reset()
                t0 = time.time()

            if step % cfg.ckpt_interval == 0:
                path = os.path.join(cfg.ckpt_dir, f"step_{step:08d}.pt")
                save_checkpoint(path, step, model, optimizer, model_cfg, cfg)
                print(f"  checkpoint -> {path}")

            if val_cache is not None and step % cfg.val_interval == 0:
                val = evaluate(model, val_cache, device, loss_kwargs)
                print(f"  [val] total {val['total']:.3f} | policy {val['policy']:.3f} "
                      f"(acc {val['acc']:.3f}) | value {val['value']:.3f} | "
                      f"aux {val['aux']:.4f}")
                t0 = time.time()

        if step >= cfg.max_steps:
            break

    path = os.path.join(cfg.ckpt_dir, f"step_{step:08d}.pt")
    save_checkpoint(path, step, model, optimizer, model_cfg, cfg)
    print(f"done at step {step}; final checkpoint -> {path}")


def parse_args():
    p = argparse.ArgumentParser(description="Train the chess network.")
    p.add_argument("--pgn", dest="pgn_path", default=TrainConfig.pgn_path)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--min-lr", type=float, default=TrainConfig.min_lr)
    p.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    p.add_argument("--shuffle-buffer", type=int, default=TrainConfig.shuffle_buffer)
    p.add_argument("--max-steps", type=int, default=TrainConfig.max_steps)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--ckpt-dir", default=TrainConfig.ckpt_dir)
    p.add_argument("--ckpt-interval", type=int, default=TrainConfig.ckpt_interval)
    p.add_argument("--log-interval", type=int, default=TrainConfig.log_interval)
    p.add_argument("--val-interval", type=int, default=TrainConfig.val_interval)
    p.add_argument("--val-batches", type=int, default=TrainConfig.val_batches)
    p.add_argument("--no-val", action="store_true", help="disable validation")
    p.add_argument("--horizon", type=int, default=ModelConfig.lookahead_n,
                   help="lookahead horizon; sets both dataset and model")
    p.add_argument("--channels", type=int, default=ModelConfig.channels)
    p.add_argument("--num-blocks", type=int, default=ModelConfig.num_blocks)
    p.add_argument("--resume", default=None)
    a = p.parse_args()

    train_cfg = TrainConfig(
        pgn_path=a.pgn_path, batch_size=a.batch_size, lr=a.lr, min_lr=a.min_lr,
        num_workers=a.num_workers, max_steps=a.max_steps, epochs=a.epochs,
        shuffle_buffer=a.shuffle_buffer,
        ckpt_dir=a.ckpt_dir, ckpt_interval=a.ckpt_interval,
        log_interval=a.log_interval, do_val=not a.no_val,
        val_interval=a.val_interval, val_batches=a.val_batches,
    )
    model_cfg = ModelConfig(channels=a.channels, num_blocks=a.num_blocks,
                            lookahead_n=a.horizon)
    return train_cfg, model_cfg, a.resume


if __name__ == "__main__":
    train_cfg, model_cfg, resume = parse_args()
    train(train_cfg, model_cfg, resume)
