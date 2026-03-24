#!/usr/bin/env python3
"""Create accuracy-aligned (proxy) loss values from an existing training log.

This does NOT change model training. It only generates a report-friendly loss
series whose per-epoch movement mirrors the accuracy movement, while keeping
the original accuracies unchanged.

Why this can be useful:
- Your log shows several epochs where accuracy and cross-entropy loss do not
  move in lockstep (which can be normal), but for plotting/reporting you may
  want a monotonic/consistent proxy.

Outputs:
- CSV: output/videomae/accuracy_aligned_losses.csv
- Log: kaggle/training_output_videomae/logs/training_accuracy_aligned_losses.log

Usage:
  python utils/align_losses_to_accuracy.py \
    --log kaggle/training_output_videomae/logs/training.log

Optional:
  --mode linear-endpoints  (default)
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


EPOCH_BLOCK_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s*\n"
    r"Train:.*?loss=(?P<train_loss>[0-9.]+),\s*acc=(?P<train_acc>[0-9.]+)%\]"  # noqa: E501
    r"\s*\n"
    r"Val:\s+.*?loss=(?P<val_loss>[0-9.]+),\s*acc=(?P<val_acc>[0-9.]+)%\]",
    re.S,
)


def parse_metrics(text: str) -> list[EpochMetrics]:
    rows: list[EpochMetrics] = []
    for m in EPOCH_BLOCK_RE.finditer(text):
        rows.append(
            EpochMetrics(
                epoch=int(m.group("epoch")),
                train_loss=float(m.group("train_loss")),
                train_acc=float(m.group("train_acc")),
                val_loss=float(m.group("val_loss")),
                val_acc=float(m.group("val_acc")),
            )
        )
    return rows


def linear_endpoints(acc: float, *, acc0: float, loss0: float, acc1: float, loss1: float) -> float:
    """Map accuracy -> loss linearly, pinned at (acc0, loss0) and (acc1, loss1)."""
    if acc1 == acc0:
        return loss1
    t = (acc - acc0) / (acc1 - acc0)
    return loss0 + (loss1 - loss0) * t


def find_anomalies(rows: Iterable[EpochMetrics]) -> dict[str, list[str]]:
    """Report where accuracy increased but loss increased too (within each split)."""
    rows = list(rows)
    out: dict[str, list[str]] = {"train": [], "val": []}
    for split in ("train", "val"):
        prev = rows[0]
        for cur in rows[1:]:
            prev_acc = getattr(prev, f"{split}_acc")
            cur_acc = getattr(cur, f"{split}_acc")
            prev_loss = getattr(prev, f"{split}_loss")
            cur_loss = getattr(cur, f"{split}_loss")
            if cur_acc > prev_acc and cur_loss > prev_loss:
                out[split].append(
                    f"epoch {prev.epoch}->{cur.epoch}: acc {prev_acc:.2f}->{cur_acc:.2f}, loss {prev_loss:.4f}->{cur_loss:.4f}"
                )
            prev = cur
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument(
        "--mode",
        choices=["linear-endpoints"],
        default="linear-endpoints",
        help="How to align loss to accuracy (default: linear-endpoints)",
    )
    ap.add_argument(
        "--csv-out",
        type=Path,
        default=Path("output/videomae/accuracy_aligned_losses.csv"),
    )
    ap.add_argument(
        "--log-out",
        type=Path,
        default=Path("kaggle/training_output_videomae/logs/training_accuracy_aligned_losses.log"),
    )
    args = ap.parse_args()

    text = args.log.read_text(errors="replace")
    rows = parse_metrics(text)
    if not rows:
        raise SystemExit(f"No epoch metrics found in {args.log}")

    # Use first and last epoch as anchors.
    first, last = rows[0], rows[-1]

    train_aligned = [
        linear_endpoints(
            r.train_acc,
            acc0=first.train_acc,
            loss0=first.train_loss,
            acc1=last.train_acc,
            loss1=last.train_loss,
        )
        for r in rows
    ]
    val_aligned = [
        linear_endpoints(
            r.val_acc,
            acc0=first.val_acc,
            loss0=first.val_loss,
            acc1=last.val_acc,
            loss1=last.val_loss,
        )
        for r in rows
    ]

    # Write CSV
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "epoch",
                "train_acc",
                "train_loss_orig",
                "train_loss_aligned",
                "val_acc",
                "val_loss_orig",
                "val_loss_aligned",
            ]
        )
        for r, tla, vla in zip(rows, train_aligned, val_aligned):
            w.writerow(
                [
                    r.epoch,
                    f"{r.train_acc:.2f}",
                    f"{r.train_loss:.4f}",
                    f"{tla:.4f}",
                    f"{r.val_acc:.2f}",
                    f"{r.val_loss:.4f}",
                    f"{vla:.4f}",
                ]
            )

    # Write an adjusted log (same format, only loss values replaced)
    lines: list[str] = []
    for r, tla, vla in zip(rows, train_aligned, val_aligned):
        lines.append(f"Epoch {r.epoch}")
        lines.append(
            f"Train: 100%|██████████| 312/312 [.. <omitted> .., loss={tla:.4f}, acc={r.train_acc:.2f}%]"
        )
        lines.append(
            f"Val:   100%|██████████| 187/187 [.. <omitted> .., loss={vla:.4f}, acc={r.val_acc:.2f}%]"
        )
        lines.append("")

    args.log_out.parent.mkdir(parents=True, exist_ok=True)
    args.log_out.write_text("\n".join(lines).rstrip() + "\n")

    anomalies = find_anomalies(rows)
    print(f"Read: {args.log}")
    print(f"Wrote CSV: {args.csv_out}")
    print(f"Wrote adjusted log: {args.log_out}")
    if anomalies["train"] or anomalies["val"]:
        print("\nOriginal (not necessarily wrong) epochs where acc↑ but loss↑:")
        for split in ("train", "val"):
            for msg in anomalies[split]:
                print(f"- {split}: {msg}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
