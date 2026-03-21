import argparse
from copy import deepcopy
import os
import sys
# Ensure repo root is on sys.path to allow imports when script is run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torch.nn import functional as F
from model import models
from unlearn_strategies import unlearn
from src import dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--root', default='./data')
    parser.add_argument('--model', required=True)            # e.g. ResNet18
    parser.add_argument('--model_path', required=True)       # path to state_dict
    parser.add_argument('--forget_idx', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cpu')           # 'cpu' or 'cuda'
    args = parser.parse_args()

    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

    # Load dataset using project's API (matches unlearn_main.py)
    train_dataset, test_dataset, num_classes, num_channels = dataset.get_dataset(
        dataset_name=args.dataset, root=args.root
    )
    full_train = list(train_dataset)
    if args.forget_idx < 0 or args.forget_idx >= len(full_train):
        raise IndexError("forget_idx out of range")

    forget_data = [full_train[args.forget_idx]]
    retain_data = [full_train[i] for i in range(len(full_train)) if i != args.forget_idx]

    # Build model (try several common constructor signatures) and load state_dict
    def build_model(name, num_classes, num_channels, device):
        ctor = getattr(models, name)
        try:
            m = ctor(num_classes=num_classes, input_channels=num_channels)
        except TypeError:
            try:
                m = ctor(num_classes=num_classes)
            except TypeError:
                try:
                    m = ctor(num_classes, num_channels)
                except Exception as e:
                    raise TypeError(f"Unable to construct model {name}: {e}")
        return m.to(device)

    print(f"Dataset size: {len(full_train)} training samples; forget_idx={args.forget_idx}")
    trained_model = build_model(args.model, num_classes, num_channels, device)
    print(f"Loading model state_dict from {args.model_path}...")
    trained_model.load_state_dict(torch.load(args.model_path, map_location=device))
    print("Loaded trained model")

    # Transform data to match model expected input channels/size (if necessary)
    def transform_sample(x, target_channels, target_size=32):
        # expect x as torch.Tensor with shape (C,H,W)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        c, h, w = x.shape
        # adjust channels
        if c != target_channels:
            if c == 1 and target_channels == 3:
                x = x.repeat(3, 1, 1)
            elif c < target_channels:
                reps = target_channels // c
                x = x.repeat(reps, 1, 1)
                if x.shape[0] < target_channels:
                    x = torch.cat([x, x[: (target_channels - x.shape[0])]], dim=0)
            else:
                x = x[:target_channels]
        # adjust size (pad or center-crop)
        if h != target_size or w != target_size:
            dh = target_size - h
            dw = target_size - w
            if dh >= 0 and dw >= 0:
                pad_top = dh // 2
                pad_bottom = dh - pad_top
                pad_left = dw // 2
                pad_right = dw - pad_left
                x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
            else:
                crop_top = (-dh) // 2
                crop_left = (-dw) // 2
                x = x[:, crop_top:crop_top + target_size, crop_left:crop_left + target_size]
        return x

    # infer target channels from model if possible
    if hasattr(trained_model, 'conv1') and hasattr(trained_model.conv1, 'in_channels'):
        target_channels = trained_model.conv1.in_channels
    else:
        target_channels = num_channels
    target_size = 32

    forget_data = [(transform_sample(x, target_channels, target_size), y) for (x, y) in forget_data]
    retain_data = [(transform_sample(x, target_channels, target_size), y) for (x, y) in retain_data]
    print(f"Transformed data to channels={target_channels}, size={target_size}")

    # Student copy to unlearn
    student = deepcopy(trained_model).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)

    # Call blindspot_unlearner (uses list-format forget_data/retain_data)
    print('Starting unlearning...')
    try:
        unlearn.blindspot_unlearner(
            model=student,
            unlearning_teacher=trained_model,
            full_trained_teacher=trained_model,
            retain_data=retain_data,
            forget_data=forget_data,
            epochs=args.epochs,
            optimizer=optimizer,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
            KL_temperature=1.0,
        )
    except Exception as e:
        import traceback, sys
        print('Error during unlearning:', e)
        traceback.print_exc(file=sys.stderr)
        raise

    out = f'checkpoint/forgot_idx_{args.forget_idx}.pt'
    os.makedirs(os.path.dirname(out) or 'checkpoint', exist_ok=True)
    torch.save(student.state_dict(), out)
    print("Saved unlearned model to", out)


if __name__ == '__main__':
    main()
