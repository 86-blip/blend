import argparse
import os
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader
import sys
# Ensure repo root is on sys.path to allow imports when script is run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import models
from src import dataset, metrics
from src import utils
import numpy as np
from torch.nn import functional as F


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


def transform_sample(x, target_channels, target_size=32):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    c, h, w = x.shape
    # channels
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
    # size
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--root', default='./data')
    parser.add_argument('--model', required=True)
    parser.add_argument('--forget_idx', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

    train_dataset, test_dataset, num_classes, num_channels = dataset.get_dataset(dataset_name=args.dataset, root=args.root)
    full_train = list(train_dataset)
    if args.forget_idx < 0 or args.forget_idx >= len(full_train):
        raise IndexError('forget_idx out of range')

    retain_data = [full_train[i] for i in range(len(full_train)) if i != args.forget_idx]

    # Build model
    model = build_model(args.model, num_classes, num_channels, device)

    # infer target channels
    if hasattr(model, 'conv1') and hasattr(model.conv1, 'in_channels'):
        target_channels = model.conv1.in_channels
    else:
        target_channels = num_channels

    target_size = 32

    # transform retain_data
    retain_data = [(transform_sample(x, target_channels, target_size), y) for (x, y) in retain_data]
    train_loader = DataLoader(retain_data, batch_size=args.batch_size, shuffle=True)
    test_list = [(transform_sample(x, target_channels, target_size), y) for (x, y) in test_dataset]
    test_loader = DataLoader(test_list, batch_size=args.batch_size, shuffle=False)

    # training setup
    if True:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_func = nn.CrossEntropyLoss().to(device)

    best_model = None
    max_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.long().to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = loss_func(out, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        mean_loss = np.mean(np.array(losses))
        train_acc = metrics.evaluate(val_loader=train_loader, model=model, device=device)['Acc']
        test_acc = metrics.evaluate(val_loader=test_loader, model=model, device=device)['Acc']
        print(f"Epoch {epoch} Loss {mean_loss:.4f} TrainAcc {train_acc} TestAcc {test_acc}")
        if test_acc >= max_test_acc:
            max_test_acc = test_acc
            best_model = copy.deepcopy(model.state_dict())

    # save best model
    out = f'checkpoint/retrain_exclude_forgetidx_{args.forget_idx}.pt'
    os.makedirs(os.path.dirname(out) or 'checkpoint', exist_ok=True)
    torch.save(best_model, out)
    print('Saved retrain-exclude model to', out)


if __name__ == '__main__':
    main()
