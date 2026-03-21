import torch
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from model import models
import numpy as np
import argparse
from tqdm import tqdm

# 1. 删除指定样本的函数
def remove_samples(dataset, forget_indices):
    """
    Removes samples with forget_indices from dataset.
    This does NOT modify the original dataset, it returns a new dataset excluding those samples.
    """
    mask = np.ones(len(dataset), dtype=bool)
    for idx in forget_indices:
        mask[idx] = False  # Mark the forget_indices as False (to exclude them)
    return Subset(dataset, np.where(mask)[0])  # Return a new Subset without the forget_indices

# 2. 重新训练模型的函数
def retrain_model(train_dataset, test_dataset, forget_indices, model_name, model_save_path, epochs=10, batch_size=64, device='cpu'):
    """
    Retrains a model after deleting samples defined by forget_indices
    """

    # 使用删除 forget_indices 样本的新的训练数据集
    modified_train_dataset = remove_samples(train_dataset, forget_indices)

    # Prepare DataLoader for new train dataset (without forget_idx)
    train_loader = DataLoader(modified_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Initialize the model
    num_classes = len(set([y for _, y in train_dataset]))  # Number of classes in the dataset
    model = models.__dict__[model_name](num_classes=num_classes).to(device)

    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        loss_list = []

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            loss_list.append(float(loss.item()))

        mean_loss = float(np.mean(np.array(loss_list))) if loss_list else 0.0

        # Reliable evaluation (weighted)
        train_acc, test_acc = evaluate(model, train_loader, test_loader, device)

        tqdm.write(
            f"[Epoch {epoch}] loss={mean_loss:.4f} "
            f"train_acc={train_acc:.4f} test_acc={test_acc:.4f} "
            f"train_ce={mean_loss:.6f} test_ce={mean_loss:.6f}"
        )

    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


def evaluate(model: torch.nn.Module, train_loader: DataLoader, test_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """
    Evaluate the model on train and test datasets.
    """
    model.eval()

    # Calculate train accuracy
    train_acc = evaluate_weighted(model, train_loader, device)["Acc"]
    test_acc = evaluate_weighted(model, test_loader, device)["Acc"]

    return train_acc, test_acc

def evaluate_weighted(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """
    True accuracy (weighted by number of samples), plus mean CE loss (weighted).
    This matches your manual check and avoids the 'mean of batch accs' pitfall.
    """
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

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


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments
    """
    ap = argparse.ArgumentParser()

    # Dataset and model arguments
    ap.add_argument("--dataset", required=True, choices=["MNist", "FMNist", "Cifar10", "Cifar100", "TinyImagenet"])
    ap.add_argument("--model", required=True)

    # Model paths
    ap.add_argument("--root", default="./data")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gpu", type=bool, default=False)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=10)

    # Forget sample indices (now comes from command-line)
    ap.add_argument("--forget_idx", type=str, required=True, help="Comma-separated list of indices to forget, e.g., 0,1,2")

    return ap.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    # Parse forget_idx into a list of integers
    forget_indices = [int(x.strip()) for x in args.forget_idx.split(",")]

    # Get dataset
    from src import dataset as ds
    train_dataset, test_dataset, num_classes, num_channels = ds.get_dataset(dataset_name=args.dataset, root=args.root)

    # Create new model save path (for retrained model)
    model_save_path = './checkpoint/forget_retrain_model.pt'

    # Retrain the model after removing forget_idx samples
    retrain_model(train_dataset, test_dataset, forget_indices, args.model, model_save_path, epochs=args.epochs, batch_size=args.batch_size, device=args.device)


if __name__ == "__main__":
    main()
