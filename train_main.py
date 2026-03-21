# train_main.py (REVISED)
import argparse
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import dataset, utils
from model import models


def build_model(model_name: str, num_classes: int, num_channels: int, device: torch.device) -> torch.nn.Module:
    """
    Robust constructor:
      - try (num_classes=..., input_channels=...)
      - then (num_classes=...)
      - then (num_classes, num_channels)
      - then no-arg
    """
    ctor = getattr(models, model_name)

    try:
        m = ctor(num_classes=num_classes, input_channels=num_channels)
        return m.to(device)
    except TypeError:
        pass

    try:
        m = ctor(num_classes=num_classes)
        return m.to(device)
    except TypeError:
        pass

    try:
        m = ctor(num_classes, num_channels)
        return m.to(device)
    except TypeError:
        pass

    m = ctor()
    return m.to(device)


@torch.no_grad()
def evaluate_weighted(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """
    True accuracy (weighted by number of samples), plus mean CE loss (weighted).
    This matches your manual check and avoids the 'mean of batch accs' pitfall.
    """
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    for batch in loader:
        x, y = batch[0].to(device), batch[1].long().to(device)
        logits = model(x)
        loss_sum += float(loss_fn(logits, y).item())
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())

    acc = 100.0 * correct / max(1, total)
    loss = loss_sum / max(1, total)
    return {"Acc": round(acc, 4), "Loss": round(loss, 6)}


def main() -> None:
    parser = argparse.ArgumentParser()

    # Device
    parser.add_argument("-gpu", type=bool, default=True, help="use gpu or not")

    # Dataset
    parser.add_argument("-root", type=str, default="./data", help="Dataset root directory")
    parser.add_argument(
        "-dataset",
        type=str,
        required=True,
        choices=["MNist", "FMNist", "Cifar10", "Cifar100", "TinyImagenet"],
        help="Dataset configuration",
    )

    # Model / Save
    parser.add_argument("-model_root", type=str, default="./checkpoint", help="Checkpoint root directory")
    parser.add_argument("-model", type=str, default="ResNet18", help="Model selection")
    parser.add_argument("-save_model", type=bool, default=True, help="Save trained model option")
    parser.add_argument(
        "-scenario",
        type=str,
        default="class",
        choices=["class", "client", "sample"],
        help="Training and unlearning scenario",
    )

    # Training hyperparameters
    parser.add_argument("-epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("-batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("-lr", type=float, default=1e-3, help="Learning rate (Adam default 1e-3 recommended)")
    parser.add_argument("-optimizer", type=str, default="adam", choices=["sgd", "adam"])
    parser.add_argument("-momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("-weight_decay", type=float, default=1e-4, help="Weight decay (Adam/SGD)")

    # Logging
    parser.add_argument("-report_training", type=bool, default=False, help="option to show training performance")
    parser.add_argument("-report_interval", type=int, default=1, help="report every N epochs")

    # Speed / reproducibility
    parser.add_argument("-seed", type=int, default=0, help="Seed for runs")
    parser.add_argument("-num_workers", type=int, default=2, help="DataLoader workers")

    args = parser.parse_args()

    # Seed
    utils.set_seed(seed=args.seed)

    # Device (keep your existing helper)
    device, device_name = utils.device_configuration(args=args)
    print(f"[Device] {device_name}")

    # Dataset (use the official pipeline from src/dataset.py)
    train_dataset, test_dataset, num_classes, num_channels = dataset.get_dataset(
        dataset_name=args.dataset,
        root=args.root,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model = build_model(args.model, num_classes=num_classes, num_channels=num_channels, device=device)

    # Optimizer
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    loss_func = nn.CrossEntropyLoss().to(device)

    max_test_acc = -1.0
    max_train_acc = -1.0
    best_model = copy.deepcopy(model)

    for epoch in tqdm(range(1, args.epochs + 1)):
        model.train()
        loss_list = []

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            loss = loss_func(output, labels)
            loss.backward()
            optimizer.step()

            loss_list.append(float(loss.item()))

        mean_loss = float(np.mean(np.array(loss_list))) if loss_list else 0.0

        # Reliable evaluation (weighted)
        train_eval = evaluate_weighted(model, train_loader, device)
        test_eval = evaluate_weighted(model, test_loader, device)
        train_acc = train_eval["Acc"]
        test_acc = test_eval["Acc"]

        if args.report_training and (epoch % max(1, args.report_interval) == 0):
            tqdm.write(
                f"[Epoch {epoch}] loss={mean_loss:.4f} "
                f"train_acc={train_acc:.4f} test_acc={test_acc:.4f} "
                f"train_ce={train_eval['Loss']:.6f} test_ce={test_eval['Loss']:.6f}"
            )

        if test_acc >= max_test_acc:
            max_test_acc = test_acc
            max_train_acc = train_acc
            best_model = copy.deepcopy(model)

    # Save
    if args.save_model:
        utils.save_model(
            model_arc=args.model,
            model=best_model,
            scenario=args.scenario,
            model_name="baseline",
            model_root=args.model_root,
            dataset_name=args.dataset,
            train_acc=max_train_acc,
            test_acc=max_test_acc,
        )

    print(f"[Best] train_acc={max_train_acc:.4f} test_acc={max_test_acc:.4f}")


if __name__ == "__main__":
    main()
