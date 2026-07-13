"""
Supervised training loop for the chess network.

Streams positions from a PGN archive (`dataset.PGNDataset`), trains the
three-head `model.ChessNet` against `loss.compute_loss`, logs the per-head loss
breakdown plus policy top-1 accuracy, and checkpoints periodically.

Run:
    python train.py --pgn pgnmentor.pgn --batch-size 256 --max-steps 100000
    python train.py --resume checkpoints/step_00010000.pt        # resume

Because the dataset is an IterableDataset there is no notion of "dataset length"
here; training is measured in optimizer steps (`--max-steps`), optionally
capped by `--epochs` full passes over the file.
"""

import argparse
import os
import time
from dataclasses import asdict, dataclass

import torch
from torch.utils.data import DataLoader

from dataset import PGNDataset
from loss import compute_loss
from model import ChessNet, ModelConfig


@dataclass
class TrainConfig:
    pgn_path: str = "pgnmentor.pgn"
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    grad_clip: float = 4.0
    num_workers: int = 4
    max_steps: int = 100_000
    epochs: int = 1                     # full passes over the PGN (upper bound)

    # loss weights (forwarded to compute_loss)
    value_weight: float = 1.0
    gamma: float = 0.85
    w_opp: float = 0.15
    w_self: float = 0.15
    label_smoothing: float = 0.0

    log_interval: int = 50
    ckpt_interval: int = 5000
    ckpt_dir: str = "checkpoints"


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup, then constant. Keeps early steps stable while BatchNorm
    statistics settle; constant afterwards is fine for a first supervised run."""
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    return cfg.lr


class Meter:
    """Running mean of the loss parts + policy accuracy since the last log."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sums = {}
        self.count = 0

    def update(self, values: dict):
        for k, v in values.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v)
        self.count += 1

    def averages(self) -> dict:
        return {k: v / max(self.count, 1) for k, v in self.sums.items()}


def save_checkpoint(path, step, model, optimizer, model_cfg, train_cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": asdict(model_cfg),
            "train_config": asdict(train_cfg),
        },
        path,
    )


def train(cfg: TrainConfig, model_cfg: ModelConfig, resume: str | None = None):
    # The dataset and the model must agree on how many horizons the aux head
    # covers, or the lookahead logits and targets silently disagree in length.
    dataset = PGNDataset(cfg.pgn_path, lookahead_horizon=model_cfg.lookahead_horizon)
    assert dataset.lookahead_horizon == model_cfg.lookahead_horizon

    device = select_device()
    use_amp = device.type == "cuda"
    print(f"device: {device} | AMP: {use_amp}")

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=cfg.num_workers > 0,
        drop_last=True,
    )

    model = ChessNet(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    start_step = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"resumed from {resume} at step {start_step}")

    model.train()
    meter = Meter()
    step = start_step
    t0 = time.time()

    for _epoch in range(cfg.epochs):
        for batch in loader:
            if step >= cfg.max_steps:
                break

            inputs = batch["input"].to(device, non_blocking=True)
            targets = {
                "policy": batch["policy"].to(device, non_blocking=True),
                "value": batch["value"].to(device, non_blocking=True),
                "lookahead": batch["lookahead"].to(device, non_blocking=True),
                "lookahead_mask": batch["lookahead_mask"].to(device, non_blocking=True),
            }

            for g in optimizer.param_groups:
                g["lr"] = lr_at(step, cfg)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
                total, parts = compute_loss(
                    outputs, targets,
                    value_weight=cfg.value_weight,
                    gamma=cfg.gamma,
                    w_opp=cfg.w_opp,
                    w_self=cfg.w_self,
                    label_smoothing=cfg.label_smoothing,
                )

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
                print(
                    f"step {step:>7} | lr {lr_at(step, cfg):.2e} | "
                    f"total {avg['total']:.3f} | policy {avg['policy']:.3f} "
                    f"(acc {avg['acc']:.3f}) | value {avg['value']:.3f} | "
                    f"aux {avg['aux']:.4f} | {rate:.0f} pos/s"
                )
                meter.reset()
                t0 = time.time()

            if step % cfg.ckpt_interval == 0:
                path = os.path.join(cfg.ckpt_dir, f"step_{step:08d}.pt")
                save_checkpoint(path, step, model, optimizer, model_cfg, cfg)
                print(f"  checkpoint -> {path}")

        if step >= cfg.max_steps:
            break

    # final checkpoint
    path = os.path.join(cfg.ckpt_dir, f"step_{step:08d}.pt")
    save_checkpoint(path, step, model, optimizer, model_cfg, cfg)
    print(f"done at step {step}; final checkpoint -> {path}")


def parse_args() -> tuple[TrainConfig, ModelConfig, str | None]:
    p = argparse.ArgumentParser(description="Train the chess network.")
    p.add_argument("--pgn", dest="pgn_path", default=TrainConfig.pgn_path)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    p.add_argument("--max-steps", type=int, default=TrainConfig.max_steps)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--ckpt-dir", default=TrainConfig.ckpt_dir)
    p.add_argument("--ckpt-interval", type=int, default=TrainConfig.ckpt_interval)
    p.add_argument("--log-interval", type=int, default=TrainConfig.log_interval)
    p.add_argument("--horizon", type=int, default=ModelConfig.lookahead_horizon,
                   help="lookahead horizon n; sets both dataset and model")
    p.add_argument("--channels", type=int, default=ModelConfig.channels)
    p.add_argument("--num-blocks", type=int, default=ModelConfig.num_blocks)
    p.add_argument("--resume", default=None)
    a = p.parse_args()

    train_cfg = TrainConfig(
        pgn_path=a.pgn_path, batch_size=a.batch_size, lr=a.lr,
        num_workers=a.num_workers, max_steps=a.max_steps, epochs=a.epochs,
        ckpt_dir=a.ckpt_dir, ckpt_interval=a.ckpt_interval,
        log_interval=a.log_interval,
    )
    model_cfg = ModelConfig(
        channels=a.channels, num_blocks=a.num_blocks, lookahead_horizon=a.horizon,
    )
    return train_cfg, model_cfg, a.resume


if __name__ == "__main__":
    train_cfg, model_cfg, resume = parse_args()
    train(train_cfg, model_cfg, resume)
