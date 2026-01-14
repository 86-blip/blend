import argparse
from copy import deepcopy
import os
import torch
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

    # Load trained model (state_dict)
    trained_model = getattr(models, args.model)(num_classes=num_classes, input_channels=num_channels).to(device)
    trained_model.load_state_dict(torch.load(args.model_path, map_location=device))

    # Student copy to unlearn
    student = deepcopy(trained_model).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)

    # Call blindspot_unlearner (uses list-format forget_data/retain_data)
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

    out = f'checkpoint/forgot_idx_{args.forget_idx}.pt'
    os.makedirs(os.path.dirname(out) or 'checkpoint', exist_ok=True)
    torch.save(student.state_dict(), out)
    print("Saved unlearned model to", out)


if __name__ == '__main__':
    main()
