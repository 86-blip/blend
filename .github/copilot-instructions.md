## Repo overview

- **Purpose**: implementations of many machine-unlearning algorithms in PyTorch (training + unlearning pipelines).
- **Entrypoints**: [train_main.py](train_main.py) (train models) and [unlearn_main.py](unlearn_main.py) (run unlearning strategies).

## Big-picture architecture

- Training pipeline: `train_main.py` uses `src.dataset.get_dataset` -> `src.raw_dataset` to load data, constructs a model via `getattr(models, args.model)`, trains and saves via `src.utils.save_model`.
- Unlearning pipeline: `unlearn_main.py` loads a saved state dict (`torch.load(args.model_path)`), splits the dataset with `src.dataset.split_unlearn_dataset`, then dispatches a strategy from `unlearn_strategies.strategies` by name (e.g., `retrain`, `scrub`, `ssd`).
- Model registry: `model/models.py` exposes factory functions (`ResNet18`, `SimpleCNN`, `MLP`) used via `getattr(models, name)`.
- Strategies registry: `unlearn_strategies/strategies.py` defines many strategy functions with a common signature; `unlearn_main.py` calls the chosen method by name with a fixed set of kwargs.

Why this matters to an AI coding assistant:
- Work typically requires editing or adding a strategy function, changing model factories, or adjusting dataset preprocessing — changes must follow the dynamic dispatch pattern (`getattr`) and the existing function signatures.

## Key patterns & conventions (project-specific)

- Dynamic dispatch by name: models and unlearning methods are referenced by string and loaded with `getattr(module, name)`. Ensure exported function/class names match CLI `-model` / `-unlearn_method` values.
- Dataset splitting: `src.dataset.split_unlearn_dataset` returns plain Python lists of `[x, y]` pairs (not always a Dataset subclass). The code constructs `DataLoader` directly from these lists (see [src/dataset.py](src/dataset.py)).
- Model save/load paths: models are saved by `src.utils.save_model` under `checkpoint/{model_arc}/{scenario}/{dataset}/` with filename pattern `{method}_{train_acc}_{test_acc}.pt`. `unlearn_main.py` expects a path to a state dict (loaded via `torch.load`). See [src/utils.py](src/utils.py).
- Device & seed: seeds are set via `src.utils.set_seed`; device selection uses `src.utils.device_configuration`. Code consistently calls `.to(device)` on models.
- Strategy signature: each strategy in `unlearn_strategies/strategies.py` accepts (args, model, unlearning_teacher, unlearn_class, unlearn_loader, retain_loader, test_loader, num_channels, num_classes, device) and returns a model. Follow this when adding new strategies.

## Typical developer workflows (commands & examples)

- Install dependencies:

```bash
pip install -r requirement.txt
```

- Train a baseline model (example):

```bash
python train_main.py -dataset Cifar10 -gpu True -epochs 30 -batch_size 128
```

- Run an unlearning method (example):

```bash
python unlearn_main.py -gpu True -dataset Cifar10 -unlearn_method retrain -model_path checkpoint/ResNet18/class/Cifar10/baseline_...pt -unlearn_class 0
```

Notes:
- The CLI uses simple argparse; some flags are non-standard (e.g., `-gpu` expects a boolean, and `-optimizer` choices in the code contain a mismatched string). Validate flag values when running or when writing automation.
- For fast iteration, run on small datasets (`MNist`) and small `-epochs` values.

## Integration points & external dependencies

- PyTorch-based project (torch 2.0 recommended in README). Check `requirement.txt` for full list.
- Many strategies depend on helper functions in `unlearn_strategies/unlearn.py` (e.g., `FGSM`, `ParameterPerturber`, `DistillKL`). When modifying strategies, inspect `unlearn_strategies/unlearn.py` first.
- Several strategies compute class-wise or per-sample statistics (NTK, fisher, ssd). These may be memory- or compute-intensive; prefer local sample debugging before large runs.

## Editing guidance for AI agents (what to change and how)

- Add a new model: define a factory in [model/models.py](model/models.py) and ensure name matches CLI `-model` value.
- Add a new strategy: implement a function in [unlearn_strategies/strategies.py](unlearn_strategies/strategies.py) following the existing signature and return a `torch.nn.Module`. Use existing helpers in `unlearn_strategies/unlearn.py` where appropriate.
- Dataset changes: update `src/raw_dataset.py` for transforms/augmentation; `src.dataset.get_dataset` delegates to raw loader.
- Saving/loading: use `src.utils.save_model` for consistent pathing; loading expects state dicts via `torch.load` — prefer `model.load_state_dict(torch.load(path))`.

## Quick diagnostics & debugging tips

- If training seems to produce no progress: verify `args.optimizer` resolves to either `sgd` or `adam` (the CLI choices string is nonstandard).
- If CUDA is not used despite `-gpu True`, inspect `src/utils.device_configuration` and `torch.cuda.is_available()`.
- For strategy-specific failures, add small reproducer inputs and run strategy function directly from an interactive script to avoid full CLI overhead.

## Files to inspect first (fast path for contributors)

- [train_main.py](train_main.py) — training entrypoint
- [unlearn_main.py](unlearn_main.py) — unlearning entrypoint
- [unlearn_strategies/strategies.py](unlearn_strategies/strategies.py) — central strategy implementations
- [model/models.py](model/models.py) and [model/resnet.py](model/resnet.py) — model factories
- [src/dataset.py](src/dataset.py) and [src/raw_dataset.py](src/raw_dataset.py) — data loading
- [src/utils.py](src/utils.py) — helpers (seed/device/save)

---
If any section is unclear or you want me to expand examples (e.g., show a minimal new strategy template or model factory), tell me which part and I will iterate.
