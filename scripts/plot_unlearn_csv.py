from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt


def read_csv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _safe_str(x: Optional[str]) -> str:
    return "" if x is None else str(x)


def get_group_key(first_row: Dict[str, str]) -> Tuple[str, str, str]:
    """
    Grouping key for de-dup:
      (mix_mode, lambda_val, block_size)
    """
    mix_mode = _safe_str(first_row.get("mix_mode", "unknown")).strip()
    lam = _safe_str(first_row.get("lambda_val", "NA")).strip()
    blk = _safe_str(first_row.get("block_size", "NA")).strip()
    return mix_mode, lam, blk


def get_label(first_row: Dict[str, str]) -> str:
    mix_mode = _safe_str(first_row.get("mix_mode", "unknown")).strip()
    lam = _safe_str(first_row.get("lambda_val", "NA")).strip()
    blk = _safe_str(first_row.get("block_size", "NA")).strip()

     # 处理重训练模型
    model_path = _safe_str(first_row.get("model_path", ""))
    if "forget_retrain_model.pt" in model_path:
        return "retrain"  # 将重训练模型命名为 "retrain"

    if mix_mode == "direct":
        return f"direct(lam={lam})"
    if mix_mode == "block":
        return f"block(bs={blk})"
    if mix_mode == "none":
        return "none"
    return mix_mode


def extract_series(rows: List[Dict[str, str]]) -> Tuple[List[int], List[float], List[float]]:
    epochs: List[int] = []
    test_acc: List[float] = []
    p_true: List[float] = []

    for r in rows:
        if r.get("phase") not in ("baseline", "train"):
            continue
        try:
            ep = int(float(r["epoch"]))
            acc = float(r["test_acc"])
            pt = float(r["p_true"])
        except Exception:
            continue
        epochs.append(ep)
        test_acc.append(acc)
        p_true.append(pt)

    return epochs, test_acc, p_true


def plot_lines(xys: List[Tuple[List[int], List[float], str]], title: str, ylabel: str, out_path: Path):
    plt.figure()
    for x, y, label in xys:
        plt.plot(x, y, label=label)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=200)
    plt.close()


def matches_filters(
    first: Dict[str, str],
    dataset: Optional[str],
    model: Optional[str],
    forget_idx: Optional[int],
    ascent_w: Optional[float],
    retain_w: Optional[float],
) -> bool:
    def eq_str(k: str, v: Optional[str]) -> bool:
        if v is None:
            return True
        return str(first.get(k, "")).lower() == str(v).lower()

    def eq_int(k: str, v: Optional[int]) -> bool:
        if v is None:
            return True
        try:
            return int(float(first.get(k, "nan"))) == int(v)
        except Exception:
            return False

    def eq_float(k: str, v: Optional[float]) -> bool:
        if v is None:
            return True
        try:
            return abs(float(first.get(k, "nan")) - float(v)) < 1e-9
        except Exception:
            return False

    return (
        eq_str("dataset", dataset)
        and eq_str("model", model)
        and eq_int("forget_idx", forget_idx)
        and eq_float("ascent_w", ascent_w)
        and eq_float("retain_w", retain_w)
    )


def main():
    ap = argparse.ArgumentParser(description="Auto-plot unlearning CSV logs (deduplicated).")
    ap.add_argument("--log_dir", default="./logs", help="directory containing CSV logs")
    ap.add_argument("--out_dir", default="./plots", help="output dir for PNGs")
    ap.add_argument("--prefix", default="unlearn", help="output filename prefix")

    # Option 1: scan by run_tag keyword (in filename)
    ap.add_argument("--run_tag", default=None, help="only include CSV files whose name contains this tag")

    # Option 2: scan by metadata filters inside CSV
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--forget_idx", type=int, default=None)
    ap.add_argument("--ascent_w", type=float, default=None)
    ap.add_argument("--retain_w", type=float, default=None)

    # Optional: restrict to specific mix modes
    ap.add_argument("--mix_modes", default=None, help="e.g. none,direct,block")

    # De-dup behavior
    ap.add_argument(
        "--dedup",
        action="store_true",
        help="deduplicate curves by (mix_mode, lambda_val, block_size) keeping the newest CSV",
    )

    # Legend verbosity
    ap.add_argument("--legend_detail", choices=["label", "file"], default="label",
                    help="legend content: 'label' or 'file' (label + filename)")

    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not log_dir.exists():
        raise FileNotFoundError(f"log_dir not found: {log_dir}")

    csv_files = sorted(log_dir.glob("*.csv"))
    if args.run_tag:
        csv_files = [p for p in csv_files if args.run_tag in p.name]

    allow_modes = None
    if args.mix_modes:
        allow_modes = set([s.strip() for s in args.mix_modes.split(",") if s.strip()])

    # Load and filter
    candidates: List[Tuple[Path, List[Dict[str, str]]]] = []
    for p in csv_files:
        rows = read_csv(p)
        if not rows:
            continue
        first = rows[0]

        if not matches_filters(first, args.dataset, args.model, args.forget_idx, args.ascent_w, args.retain_w):
            continue

        if allow_modes is not None:
            mm = str(first.get("mix_mode", "")).strip()
            if mm not in allow_modes:
                continue

        candidates.append((p, rows))

    if len(candidates) == 0:
        raise RuntimeError(
            f"No CSV matched. Scanned: {log_dir}\n"
            f"Try: --run_tag <tag> or relax filters."
        )

    # Deduplicate (keep newest file per group key)
    picked: List[Tuple[Path, List[Dict[str, str]]]] = []
    if args.dedup:
        best: Dict[Tuple[str, str, str], Tuple[Path, List[Dict[str, str]]]] = {}
        for p, rows in candidates:
            key = get_group_key(rows[0])
            if key not in best:
                best[key] = (p, rows)
            else:
                # compare modification time
                if p.stat().st_mtime > best[key][0].stat().st_mtime:
                    best[key] = (p, rows)
        picked = list(best.values())
    else:
        picked = candidates

    # Build plot series
    series_acc: List[Tuple[List[int], List[float], str]] = []
    series_ptr: List[Tuple[List[int], List[float], str]] = []
    picked_names = []

    # Stable ordering: none, direct, block first if present
    def sort_key(item):
        p, rows = item
        mm = str(rows[0].get("mix_mode", ""))
        order = {"none": 0, "direct": 1, "block": 2}
        return (order.get(mm, 99), p.name)

    picked = sorted(picked, key=sort_key)

    for p, rows in picked:
        first = rows[0]
        label = get_label(first)

        if args.legend_detail == "file":
            label = f"{label} | {p.name}"

        epochs, acc, ptrue = extract_series(rows)
        series_acc.append((epochs, acc, label))
        series_ptr.append((epochs, ptrue, label))
        picked_names.append(p.name)

    acc_path = out_dir / f"{args.prefix}_test_acc.png"
    ptr_path = out_dir / f"{args.prefix}_p_true.png"

    plot_lines(series_acc, title="Test Accuracy vs Epoch", ylabel="test_acc (%)", out_path=acc_path)
    plot_lines(series_ptr, title="Forget Sample p_true vs Epoch", ylabel="p_true", out_path=ptr_path)

    print("[Picked CSV] (after filtering" + (", dedup" if args.dedup else "") + ")")
    for n in picked_names:
        print(" -", n)
    print(f"[Saved] {acc_path}")
    print(f"[Saved] {ptr_path}")


if __name__ == "__main__":
    main()
