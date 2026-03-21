import argparse
import torch
from torch.utils.data import DataLoader

from src import dataset as ds
from model import models


def instantiate_model(model_name: str, num_classes: int, num_channels: int):
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
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["MNist", "FMNist", "Cifar10", "Cifar100", "TinyImagenet"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--root", default="./data")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device(args.device)

    train_ds, test_ds, num_classes, num_channels = ds.get_dataset(dataset_name=args.dataset, root=args.root)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = instantiate_model(args.model, num_classes=num_classes, num_channels=num_channels).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    correct = 0
    total = 0

    for i, batch in enumerate(loader):
        x, y = batch[0].to(device), batch[1].to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
        if i < 2:
            print(f"batch {i}: correct {(pred==y).sum().item()} / {y.numel()}")

    acc = 100.0 * correct / total
    print(f"MANUAL test_acc = {acc:.4f}%")


if __name__ == "__main__":
    main()
