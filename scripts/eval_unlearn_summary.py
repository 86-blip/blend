from __future__ import annotations

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src import dataset as ds
from src import utils
from model import models


def _make_device(device_str: str, gpu_flag: bool) -> torch.device:
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if gpu_flag and torch.cuda.is_available() else "cpu")


def _instantiate_model(model_name: str, num_classes: int, num_channels: int) -> torch.nn.Module:
    ctor = getattr(models, model_name)
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
def eval_accuracy(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.long().to(device)
        pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return 100.0 * correct / max(1, total)


@torch.no_grad()
def eval_forget_metrics(
    model: torch.nn.Module,
    dataset,
    forget_indices: List[int],
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    correct = 0
    p_trues = []
    ces = []
    for idx in forget_indices:
        x, y = dataset[int(idx)]
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        x = x.unsqueeze(0).to(device)
        y_t = torch.tensor([int(y)], dtype=torch.long, device=device)

        logits = model(x)
        probs = F.softmax(logits, dim=1)
        pred = int(torch.argmax(probs, dim=1).item())
        correct += int(pred == int(y))

        p_true = float(probs[0, int(y)].item())
        p_trues.append(p_true)

        ce = float(F.cross_entropy(logits, y_t).item())
        ces.append(ce)

    forget_acc = 100.0 * correct / max(1, len(forget_indices))
    return {
        "forget_acc": forget_acc,
        "forget_p_true_mean": float(np.mean(p_trues)) if p_trues else float("nan"),
        "forget_ce_mean": float(np.mean(ces)) if ces else float("nan"),
    }


@torch.no_grad()
def per_sample_losses(
    model: torch.nn.Module,
    dataset,
    indices: List[int],
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 2,
) -> np.ndarray:
    sub = Subset(dataset, indices)
    loader = DataLoader(sub, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model.eval()
    losses = []
    for x, y in loader:
        x = x.to(device)
        y = y.long().to(device)
        logits = model(x)
        l = F.cross_entropy(logits, y, reduction="none")
        losses.append(l.detach().cpu().numpy())
    if not losses:
        return np.array([], dtype=np.float32)
    return np.concatenate(losses, axis=0).astype(np.float32)


def mia_best_threshold_acc(member_losses: np.ndarray, nonmember_losses: np.ndarray) -> Tuple[float, float]:
    all_losses = np.concatenate([member_losses, nonmember_losses], axis=0)
    if all_losses.size == 0:
        return float("nan"), float("nan")

    thresholds = np.unique(all_losses)

    best_acc = -1.0
    best_tau = float(thresholds[0])

    m = member_losses
    n = nonmember_losses

    for tau in thresholds:
        m_pred = (m <= tau).astype(np.int32)
        n_pred = (n <= tau).astype(np.int32)

        correct = int((m_pred == 1).sum()) + int((n_pred == 0).sum())
        total = int(m.size + n.size)
        acc = correct / max(1, total)

        if acc > best_acc:
            best_acc = acc
            best_tau = float(tau)

    return float(best_acc), float(best_tau)


def mia_auc(member_losses: np.ndarray, nonmember_losses: np.ndarray) -> float:
    m = member_losses
    n = nonmember_losses
    if m.size == 0 or n.size == 0:
        return float("nan")

    n_sorted = np.sort(n)
    counts = (n.size - np.searchsorted(n_sorted, m, side="right")).astype(np.float64)
    auc = float(counts.sum() / (m.size * n.size))
    return auc


def sample_indices_for_mia(
    train_dataset,
    test_dataset,
    forget_indices: List[int],
    seed: int,
    n_member: int,
    n_nonmember: int,
    exclude_forget: bool,
) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)

    train_pool = np.arange(len(train_dataset), dtype=np.int64)
    if exclude_forget:
        mask = np.ones(len(train_pool), dtype=bool)
        for fi in forget_indices:
            if 0 <= int(fi) < len(train_pool):
                mask[int(fi)] = False
        train_pool = train_pool[mask]

    test_pool = np.arange(len(test_dataset), dtype=np.int64)

    n_member = min(n_member, len(train_pool))
    n_nonmember = min(n_nonmember, len(test_pool))

    member_idx = rng.choice(train_pool, size=n_member, replace=False).tolist()
    nonmember_idx = rng.choice(test_pool, size=n_nonmember, replace=False).tolist()
    return member_idx, nonmember_idx


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", required=True, choices=["MNist", "FMNist", "Cifar10", "Cifar100", "TinyImagenet"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--baseline_paths", required=True, help="baseline checkpoint paths (comma separated)")
    ap.add_argument("--unlearn_paths", required=True, help="unlearned checkpoint paths (comma separated)")
    ap.add_argument("--retrain_model", required=True, help="retrained model checkpoint path")
    ap.add_argument("--forget_idx", type=str, required=True, help="e.g. 0 or 0,1,2")
    ap.add_argument("--root", default="./data")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gpu", type=bool, default=False)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--mia_member_n", type=int, default=2000)
    ap.add_argument("--mia_nonmember_n", type=int, default=2000)
    ap.add_argument("--mia_exclude_forget", action="store_true")

    return ap.parse_args()


def load_model(ckpt_path: str, model_name: str, num_classes: int, num_channels: int, device: torch.device) -> torch.nn.Module:
    model = _instantiate_model(model_name, num_classes=num_classes, num_channels=num_channels).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def summarize_one(
    tag: str,
    model: torch.nn.Module,
    train_dataset,
    test_dataset,
    forget_indices: List[int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    mia_member_idx: List[int],
    mia_nonmember_idx: List[int],
) -> Dict[str, float]:
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=num_workers)
    retain_acc = eval_accuracy(model, test_loader, device)

    f = eval_forget_metrics(model, train_dataset, forget_indices, device)

    member_losses = per_sample_losses(model, train_dataset, mia_member_idx, device, batch_size=batch_size, num_workers=num_workers)
    nonmember_losses = per_sample_losses(model, test_dataset, mia_nonmember_idx, device, batch_size=batch_size, num_workers=num_workers)
    mia_acc, tau = mia_best_threshold_acc(member_losses, nonmember_losses)
    auc = mia_auc(member_losses, nonmember_losses)

    return {
        "model_type": tag,
        "retain_acc": retain_acc,
        "forget_acc": f["forget_acc"],
        "forget_p_true_mean": f["forget_p_true_mean"],
        "forget_ce_mean": f["forget_ce_mean"],
        "mia_acc": 100.0 * mia_acc,
        "mia_auc": auc,
        "mia_tau": tau,
    }


def print_table(rows: List[Dict[str, float]]):
    headers = ["模型类型", "Retain Acc(%)", "Forget Acc(%)", "Forget p_true(mean)", "MIA Acc(%)", "MIA AUC"]
    print("\n总结表格：")
    print("{:<25s} {:>14s} {:>14s} {:>18s} {:>12s} {:>10s}".format(*headers))
    for r in rows:
        print(
            "{:<25s} {:>14.4f} {:>14.4f} {:>18.6f} {:>12.2f} {:>10.4f}".format(
                str(r["model_type"]),
                float(r["retain_acc"]),
                float(r["forget_acc"]),
                float(r["forget_p_true_mean"]),
                float(r["mia_acc"]),
                float(r["mia_auc"]),
            )
        )
    print("")


def main() -> None:
    args = parse_args()
    utils.set_seed(seed=args.seed)
    device = _make_device(args.device, args.gpu)
    print(f"[Device] {device}")

    forget_indices = [int(x.strip()) for x in args.forget_idx.split(",") if x.strip() != ""]

    train_dataset, test_dataset, num_classes, num_channels = ds.get_dataset(dataset_name=args.dataset, root=args.root)

    mia_member_idx, mia_nonmember_idx = sample_indices_for_mia(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        forget_indices=forget_indices,
        seed=args.seed,
        n_member=args.mia_member_n,
        n_nonmember=args.mia_nonmember_n,
        exclude_forget=args.mia_exclude_forget,
    )

    rows = []
    baseline_paths = args.baseline_paths.split(',')
    unlearn_paths = args.unlearn_paths.split(',')

    # For baseline model
    for baseline_path in baseline_paths:
        base_model = load_model(baseline_path, args.model, num_classes, num_channels, device)
        base_row = summarize_one(
            tag=f"基准模型 - {baseline_path}",
            model=base_model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            forget_indices=forget_indices,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            mia_member_idx=mia_member_idx,
            mia_nonmember_idx=mia_nonmember_idx,
        )
        rows.append(base_row)

    # For unlearn models
    for unlearn_path in unlearn_paths:
        unlearn_model = load_model(unlearn_path, args.model, num_classes, num_channels, device)
        unlearn_row = summarize_one(
            tag=f"去忘模型 - {unlearn_path}",
            model=unlearn_model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            forget_indices=forget_indices,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            mia_member_idx=mia_member_idx,
            mia_nonmember_idx=mia_nonmember_idx,
        )
        rows.append(unlearn_row)

    # For retrained model
    retrain_model = load_model(args.retrain_model, args.model, num_classes, num_channels, device)
    retrain_row = summarize_one(
        tag=f"重训练模型 - {args.retrain_model}",
        model=retrain_model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        forget_indices=forget_indices,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mia_member_idx=mia_member_idx,
        mia_nonmember_idx=mia_nonmember_idx,
    )
    rows.append(retrain_row)

    print_table(rows)


if __name__ == "__main__":
    main()
