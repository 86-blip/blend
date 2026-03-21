import argparse
import sys
sys.path.insert(0, '.')
from model import models
from src import dataset, metrics
from torch.utils.data import DataLoader
import torch
from torch.nn import functional as F
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--root', default='./data')
    parser.add_argument('--model', required=True)
    parser.add_argument('--orig', required=True, help='原始模型 state_dict 路径')
    parser.add_argument('--unlearn', required=True, help='去忘后模型 state_dict 路径')
    parser.add_argument('--forget_idx', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

    train_dataset, test_dataset, num_classes, num_channels = dataset.get_dataset(dataset_name=args.dataset, root=args.root)
    full_train = list(train_dataset)
    if args.forget_idx < 0 or args.forget_idx >= len(full_train):
        raise IndexError('forget_idx out of range')

    forget_data = [full_train[args.forget_idx]]
    retain_data = [full_train[i] for i in range(len(full_train)) if i != args.forget_idx]

    # DataLoaders will be created after we construct the model so we can
    # match input channels/size to the model's expected format.
    retain_loader = None
    unlearn_loader = None
    test_loader = None

    # load models (try several common constructor signatures)
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

    orig_model = build_model(args.model, num_classes, num_channels, device)
    unlearn_model = build_model(args.model, num_classes, num_channels, device)
    orig_model.load_state_dict(torch.load(args.orig, map_location=device))
    unlearn_model.load_state_dict(torch.load(args.unlearn, map_location=device))

    # Transform samples to match model input (channels / size)
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
        # size (pad or crop)
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

    # infer target channels / size from model if possible
    if hasattr(orig_model, 'conv1') and hasattr(orig_model.conv1, 'in_channels'):
        target_channels = orig_model.conv1.in_channels
    else:
        target_channels = num_channels
    target_size = 32

    forget_data = [(transform_sample(x, target_channels, target_size), y) for (x, y) in forget_data]
    retain_data = [(transform_sample(x, target_channels, target_size), y) for (x, y) in retain_data]
    test_list = [(transform_sample(x, target_channels, target_size), y) for (x, y) in test_dataset]

    print(f"Transformed evaluation data to channels={target_channels}, size={target_size}")

    retain_loader = DataLoader(retain_data, batch_size=args.batch_size, shuffle=False)
    unlearn_loader = DataLoader(forget_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=args.batch_size, shuffle=False)

    print('Evaluating original model...')
    orig_retain = metrics.evaluate(val_loader=retain_loader, model=orig_model, device=device)['Acc']
    orig_unlearn = metrics.evaluate(val_loader=unlearn_loader, model=orig_model, device=device)['Acc']
    orig_mia = metrics.mia(retain_loader=retain_loader, forget_loader=unlearn_loader, test_loader=test_loader, model=orig_model)

    print('Evaluating unlearned model...')
    un_retain = metrics.evaluate(val_loader=retain_loader, model=unlearn_model, device=device)['Acc']
    un_unlearn = metrics.evaluate(val_loader=unlearn_loader, model=unlearn_model, device=device)['Acc']
    un_mia = metrics.mia(retain_loader=retain_loader, forget_loader=unlearn_loader, test_loader=test_loader, model=unlearn_model)

    print('\nSummary:')
    print(f"Original - Retain Acc: {orig_retain}  Forget Acc: {orig_unlearn}  MIA: {orig_mia}")
    print(f"Unlearned - Retain Acc: {un_retain}  Forget Acc: {un_unlearn}  MIA: {un_mia}")

    # file info
    if os.path.exists(args.unlearn):
        st = os.stat(args.unlearn)
        print(f"Saved unlearned model: {args.unlearn} (size={st.st_size} bytes)")
    else:
        print('Unlearned model file not found')


if __name__ == '__main__':
    main()
