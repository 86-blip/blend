# src/protect_selection.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F


@dataclass
class SampleScore:
    idx: int
    y: int
    loss: float
    margin: float
    p_top1: float
    p_top2: float
    pred: int


class IndexDataset(torch.utils.data.Dataset):
    """Wrap dataset to return (x, y, idx, *rest)."""

    def __init__(self, base_ds: torch.utils.data.Dataset):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx: int):
        item = self.base_ds[idx]
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            x, y = item[0], item[1]
            rest = item[2:]
            return (x, y, idx, *rest)
        raise ValueError("Dataset item must return at least (x, y).")


@torch.no_grad()
def score_dataset(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> List[SampleScore]:
    model.eval()
    scores: List[SampleScore] = []

    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].long().to(device)
        idx = batch[2]

        if torch.is_tensor(idx):
            idx_list = idx.detach().cpu().tolist()
        else:
            idx_list = list(idx)

        logits = model(x)

        loss_vec = F.cross_entropy(logits, y, reduction="none").detach().cpu().tolist()

        probs = F.softmax(logits, dim=1)
        top2 = torch.topk(probs, k=2, dim=1)
        p1 = top2.values[:, 0]
        p2 = top2.values[:, 1]

        margin = (p1 - p2).detach().cpu().tolist()
        p1l = p1.detach().cpu().tolist()
        p2l = p2.detach().cpu().tolist()

        pred = torch.argmax(probs, dim=1).detach().cpu().tolist()
        y_cpu = y.detach().cpu().tolist()

        for i in range(len(idx_list)):
            scores.append(
                SampleScore(
                    idx=int(idx_list[i]),
                    y=int(y_cpu[i]),
                    loss=float(loss_vec[i]),
                    margin=float(margin[i]),
                    p_top1=float(p1l[i]),
                    p_top2=float(p2l[i]),
                    pred=int(pred[i]),
                )
            )

    return scores


def _group_by_class(scores: Sequence[SampleScore]) -> Dict[int, List[SampleScore]]:
    by: Dict[int, List[SampleScore]] = {}
    for s in scores:
        by.setdefault(s.y, []).append(s)
    return by


def select_protect_indices(
    scores: Sequence[SampleScore],
    per_class_k: int,
    strategy: str = "margin_mix",
    boundary_frac: float = 0.7,
    exclude_classes: Optional[Iterable[int]] = None,
) -> Dict[int, List[int]]:
    """
    strategy:
      - margin     : smallest margin (closest to boundary)
      - loss       : largest CE loss (hard)
      - error      : misclassified first, then by smallest margin
      - margin_mix : boundary_frac*K smallest margin + rest largest margin
    """
    if per_class_k <= 0:
        raise ValueError("per_class_k must be > 0")

    ex = set(exclude_classes or [])
    filtered = [s for s in scores if s.y not in ex]
    by_class = _group_by_class(filtered)

    selected: Dict[int, List[int]] = {}

    for y, items in by_class.items():
        if not items:
            continue
        k = min(per_class_k, len(items))

        if strategy == "margin":
            chosen = sorted(items, key=lambda s: s.margin)[:k]

        elif strategy == "loss":
            chosen = sorted(items, key=lambda s: s.loss, reverse=True)[:k]

        elif strategy == "error":
            wrong = [s for s in items if s.pred != s.y]
            right = [s for s in items if s.pred == s.y]
            wrong = sorted(wrong, key=lambda s: s.margin)
            right = sorted(right, key=lambda s: s.margin)
            chosen = (wrong + right)[:k]

        elif strategy == "margin_mix":
            if not (0.0 <= boundary_frac <= 1.0):
                raise ValueError("boundary_frac must be in [0,1].")

            kb = int(round(k * boundary_frac))
            kb = max(0, min(kb, k))
            ke = k - kb

            items_sorted = sorted(items, key=lambda s: s.margin)  # small -> boundary
            boundary = items_sorted[:kb]
            easy = list(reversed(items_sorted))[:ke]  # large margin -> representative
            chosen = boundary + easy

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # unique indices
        uniq: List[int] = []
        seen = set()
        for s in chosen:
            if s.idx not in seen:
                uniq.append(s.idx)
                seen.add(s.idx)

        selected[int(y)] = uniq

    return selected
