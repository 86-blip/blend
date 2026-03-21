# scripts/forget_one_blend.py
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import json
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src import dataset as ds
from src import metrics, utils
from model import models
from src.protect_selection import IndexDataset, score_dataset, select_protect_indices


def _make_device(device_str: Optional[str], gpu_flag: bool) -> torch.device:
    if device_str is not None:
        return torch.device(device_str)
    return torch.device("cuda" if gpu_flag and torch.cuda.is_available() else "cpu")


def _instantiate_model(model_name: str, num_classes: int, num_channels: int) -> torch.nn.Module:
    """
    Make this robust to different model constructors.
    - Some models accept (num_classes, input_channels)
    - Some accept only (num_classes)
    - Some accept no args
    """
    ctor = getattr(models, model_name)

    # Try common signatures
    try:
        return ctor(num_classes=num_classes, input_channels=num_channels)
    except TypeError:
        pass
    try:
        return ctor(num_classes=num_classes)
    except TypeError:
        pass
    return ctor()


@torch.no_grad()
def eval_forget_sample(model: torch.nn.Module, x_one: torch.Tensor, y_one: torch.Tensor):
    """
    Evaluate forget sample:
      pred, p_true, p_max, margin(top1-top2)
    x_one: [1,C,H,W]
    y_one: [1]
    """
    model.eval()
    logits = model(x_one)
    probs = F.softmax(logits, dim=1).squeeze(0)  # [num_classes]

    pred = int(torch.argmax(probs).item())
    y = int(y_one.item())

    p_true = float(probs[y].item())
    p_max = float(torch.max(probs).item())

    top2 = torch.topk(probs, k=2)
    margin = float((top2.values[0] - top2.values[1]).item())

    model.train()
    return pred, p_true, p_max, margin


def direct_mix(x: torch.Tensor, xb: torch.Tensor, lam: float) -> torch.Tensor:
    return lam * x + (1.0 - lam) * xb


def block_mix(x: torch.Tensor, xb: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    Patch-wise mixing on tensor [B,C,H,W] with same shape.
    Alternate patches between x and xb in a checker-like pattern.
    """
    assert x.shape == xb.shape
    b, c, h, w = x.shape
    out = x.clone()

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            use_xb = ((i // block_size) + (j // block_size)) % 2 == 1
            if use_xb:
                out[:, :, i:i + block_size, j:j + block_size] = xb[:, :, i:i + block_size, j:j + block_size]
    return out


def build_or_load_protect_set(
    *,
    baseline_model: torch.nn.Module,
    train_dataset,
    dataset_name: str,
    model_name: str,
    model_path: str,
    device: torch.device,
    per_class_k: int,
    strategy: str,
    boundary_frac: float,
    exclude_class: Optional[int],
    out_dir: str,
    batch_size: int = 256,
    num_workers: int = 2,
) -> np.ndarray:
    """
    Returns a flattened np.int64 array of protect indices.
    Will cache to out_dir for reuse.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    excl = f"_excl{exclude_class}" if exclude_class is not None else ""
    tag = f"{dataset_name}_{model_name}{excl}_k{per_class_k}_{strategy}"
    npy_file = out_path / f"protect_{tag}.npy"
    json_file = out_path / f"protect_{tag}.json"

    if npy_file.exists():
        return np.load(npy_file)

    idx_ds = IndexDataset(train_dataset)
    loader = DataLoader(
        idx_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    scores = score_dataset(model=baseline_model, loader=loader, device=device)
    selected_by_class = select_protect_indices(
        scores=scores,
        per_class_k=per_class_k,
        strategy=strategy,
        boundary_frac=boundary_frac,
        exclude_classes=[exclude_class] if exclude_class is not None else None,
    )

    flat = []
    for y in sorted(selected_by_class.keys()):
        flat.extend(selected_by_class[y])

    flat_arr = np.array(flat, dtype=np.int64)
    np.save(npy_file, flat_arr)

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in selected_by_class.items()}, f, indent=2)

    print(f"[xb] saved: {npy_file} (n={len(flat_arr)})")
    return flat_arr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", required=True, choices=["MNist", "FMNist", "Cifar10", "Cifar100", "TinyImagenet"])
    p.add_argument("--model", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--forget_idx", type=int, required=True)

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)

    # device controls (compatible)
    p.add_argument("--device", type=str, default=None, help="e.g. cpu / cuda / cuda:0")
    p.add_argument("--gpu", type=bool, default=True)

    p.add_argument("--root", type=str, default="./data")

    # xb selection config
    p.add_argument("--protect_out_dir", type=str, default="./data/protect_sets")
    p.add_argument("--xb_per_class_k", type=int, default=200)
    p.add_argument("--xb_strategy", type=str, default="margin_mix", choices=["margin", "loss", "error", "margin_mix"])
    p.add_argument("--xb_boundary_frac", type=float, default=0.7)
    p.add_argument("--exclude_forget_class", action="store_true", help="exclude forget sample's class from xb")

    # training objective weights
    p.add_argument("--ascent_w", type=float, default=1.0, help="weight for forget loss ascent")
    p.add_argument("--retain_w", type=float, default=1.0, help="weight for xb retain loss")

    # optional mixing injection
    p.add_argument("--mix_mode", type=str, default="none", choices=["none", "direct", "block"])
    p.add_argument("--lambda_val", type=float, default=0.5)
    p.add_argument("--block_size", type=int, default=4)

    # save
    p.add_argument("--save_path", type=str, default=None, help="where to save unlearned model")

    # speed
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)

    # CSV logging
    p.add_argument("--csv_path", type=str, default=None, help="optional explicit CSV path to save logs")
    p.add_argument("--run_tag", type=str, default="", help="suffix tag appended to auto CSV filename")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    utils.set_seed(seed=args.seed)

    device = _make_device(args.device, args.gpu)
    print(f"[Device] {device}")

    train_dataset, test_dataset, num_classes, num_channels = ds.get_dataset(dataset_name=args.dataset, root=args.root)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=args.num_workers)

    # load baseline model
    model = _instantiate_model(args.model, num_classes=num_classes, num_channels=num_channels).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.train()

    # fetch forget sample
    x_forget, y_forget = train_dataset[args.forget_idx]
    if not torch.is_tensor(x_forget):
        x_forget = torch.tensor(x_forget)
    x_forget = x_forget.unsqueeze(0).to(device)  # [1,C,H,W]
    y_forget = torch.tensor([int(y_forget)], dtype=torch.long, device=device)

    forget_class = int(y_forget.item())
    exclude_class = forget_class if args.exclude_forget_class else None
    print(f"[Forget] idx={args.forget_idx} class={forget_class} exclude_class={exclude_class}")

    # ===== Baseline evaluation (before unlearning) =====
    model.eval()
    base_metrics = metrics.evaluate(val_loader=test_loader, model=model, device=device)
    base_acc = base_metrics["Acc"]
    base_pred, base_p_true, base_p_max, base_margin = eval_forget_sample(model, x_forget, y_forget)
    print(
        f"[Baseline] test_acc={base_acc} "
        f"forget_true={int(y_forget.item())} forget_pred={base_pred} "
        f"p_true={base_p_true:.6f} p_max={base_p_max:.6f} margin={base_margin:.6f}"
    )
    model.train()
    # ================================================

    # ===== CSV logger init =====
    csv_path = args.csv_path
    if csv_path is None:
        out_dir = Path("./logs")
        out_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = f"_{args.run_tag}" if args.run_tag else ""
        csv_path = str(
            out_dir / (
                f"unlearn_{args.dataset}_{args.model}"
                f"_idx{args.forget_idx}"
                f"_mix{args.mix_mode}"
                f"_lam{args.lambda_val if args.mix_mode=='direct' else 'NA'}"
                f"_blk{args.block_size if args.mix_mode=='block' else 'NA'}"
                f"_aw{args.ascent_w}_rw{args.retain_w}"
                f"_ep{args.epochs}_bs{args.batch_size}"
                f"{tag}_{ts}.csv"
            )
        )

    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "phase", "epoch",
            "dataset", "model", "model_path",
            "forget_idx", "forget_true",
            "mix_mode", "lambda_val", "block_size",
            "ascent_w", "retain_w", "lr", "batch_size",
            "test_acc",
            "forget_pred", "p_true", "p_max", "margin",
            "CE_forget", "CE_retain", "total",
        ],
    )
    csv_writer.writeheader()

    csv_writer.writerow({
        "phase": "baseline",
        "epoch": 0,
        "dataset": args.dataset,
        "model": args.model,
        "model_path": args.model_path,
        "forget_idx": args.forget_idx,
        "forget_true": int(y_forget.item()),
        "mix_mode": args.mix_mode,
        "lambda_val": args.lambda_val,
        "block_size": args.block_size,
        "ascent_w": args.ascent_w,
        "retain_w": args.retain_w,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "test_acc": base_acc,
        "forget_pred": base_pred,
        "p_true": base_p_true,
        "p_max": base_p_max,
        "margin": base_margin,
        "CE_forget": "",
        "CE_retain": "",
        "total": "",
    })
    csv_file.flush()
    print(f"[CSV] {csv_path}")
    # ==========================

    try:
        # build/load protect indices
        protect_idx = build_or_load_protect_set(
            baseline_model=model,
            train_dataset=train_dataset,
            dataset_name=args.dataset,
            model_name=args.model,
            model_path=args.model_path,
            device=device,
            per_class_k=args.xb_per_class_k,
            strategy=args.xb_strategy,
            boundary_frac=args.xb_boundary_frac,
            exclude_class=exclude_class,
            out_dir=args.protect_out_dir,
            batch_size=256,
            num_workers=args.num_workers,
        )
        if len(protect_idx) == 0:
            raise RuntimeError("protect_idx is empty. Check xb selection settings.")

        # optimizer
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        # training loop
        model.train()
        for ep in range(1, args.epochs + 1):
            steps = max(1, 200 // max(1, args.batch_size))
            loss_log = []

            for _ in range(steps):
                xb_ids = np.random.choice(protect_idx, size=args.batch_size, replace=(len(protect_idx) < args.batch_size))
                xb_list = [train_dataset[int(i)] for i in xb_ids]
                xb = torch.stack(
                    [t[0] if torch.is_tensor(t[0]) else torch.tensor(t[0]) for t in xb_list],
                    dim=0
                ).to(device)
                yb = torch.tensor([int(t[1]) for t in xb_list], dtype=torch.long, device=device)

                xf = x_forget.expand(args.batch_size, *x_forget.shape[1:])

                if args.mix_mode == "direct":
                    x_inj = direct_mix(xf, xb, lam=args.lambda_val)
                elif args.mix_mode == "block":
                    x_inj = block_mix(xf, xb, block_size=args.block_size)
                else:
                    x_inj = xf

                opt.zero_grad()

                logits_forget = model(x_inj)
                ce_forget = F.cross_entropy(logits_forget, y_forget.expand(args.batch_size))

                logits_retain = model(xb)
                ce_retain = F.cross_entropy(logits_retain, yb)

                total = args.ascent_w * (-ce_forget) + args.retain_w * ce_retain
                total.backward()
                opt.step()

                loss_log.append((ce_forget.item(), ce_retain.item(), total.item()))

            # eval
            model.eval()
            test_acc = metrics.evaluate(val_loader=test_loader, model=model, device=device)["Acc"]
            cur_pred, cur_p_true, cur_p_max, cur_margin = eval_forget_sample(model, x_forget, y_forget)
            model.train()

            avg = np.mean(np.array(loss_log), axis=0)
            print(
                f"[Epoch {ep}] "
                f"CE_forget={avg[0]:.4f} CE_retain={avg[1]:.4f} total={avg[2]:.4f} "
                f"test_acc={test_acc} "
                f"forget_true={int(y_forget.item())} forget_pred={cur_pred} "
                f"p_true={cur_p_true:.6f} p_max={cur_p_max:.6f} margin={cur_margin:.6f}"
            )

            csv_writer.writerow({
                "phase": "train",
                "epoch": ep,
                "dataset": args.dataset,
                "model": args.model,
                "model_path": args.model_path,
                "forget_idx": args.forget_idx,
                "forget_true": int(y_forget.item()),
                "mix_mode": args.mix_mode,
                "lambda_val": args.lambda_val,
                "block_size": args.block_size,
                "ascent_w": args.ascent_w,
                "retain_w": args.retain_w,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "test_acc": test_acc,
                "forget_pred": cur_pred,
                "p_true": cur_p_true,
                "p_max": cur_p_max,
                "margin": cur_margin,
                "CE_forget": float(avg[0]),
                "CE_retain": float(avg[1]),
                "total": float(avg[2]),
            })
            csv_file.flush()

        # save
        save_path = args.save_path
        if save_path is None:
            ckpt_dir = Path("./checkpoint") / args.dataset / args.model
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            # mix info
            if args.mix_mode == "direct":
                mix_tag = f"direct_lam{args.lambda_val}"
            elif args.mix_mode == "block":
                mix_tag = f"block_bs{args.block_size}"
            else:
                mix_tag = "none"

            save_name = (
                f"forget_idx{args.forget_idx}_"
                f"mix{mix_tag}_"
                f"aw{args.ascent_w}_rw{args.retain_w}_"
                f"ep{args.epochs}_bs{args.batch_size}.pt"
            )
            save_path = str(ckpt_dir / save_name)

        torch.save(model.state_dict(), save_path)
        print(f"[Saved] {save_path}")


    finally:
        csv_file.close()


if __name__ == "__main__":
    main()
