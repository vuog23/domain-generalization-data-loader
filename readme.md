# Data Loader Usage

This project provides three DomainBed-style loader builders in `src/data/dataloader.py`:

- `MultiDomainLoaderBuilder` for grouped multi-domain datasets such as DomainNet and NICOPP.
- `SingleDomainLoaderBuilder` for datasets where each domain folder directly contains class folders.
- `SplitDomainLoaderBuilder` for datasets where each domain contains split folders such as `train` and `test`. VLCS uses this builder.

The examples below are based on `notebooks/data_loader_usage.ipynb`.

## Setup

Run notebooks from the project root, or move one level up when running from the `notebooks/` directory:

```python
import os
from pathlib import Path

if Path.cwd().name == "notebooks":
    os.chdir("..")

PROJECT_ROOT = Path.cwd()
```

Import the loader builders and transforms:

```python
from src.data.dataloader import (
    MultiDomainLoaderBuilder,
    SingleDomainLoaderBuilder,
    SplitDomainLoaderBuilder,
)
from src.data.transforms import get_source_transform, get_eval_transform
```

Each builder returns the same five objects:

```python
(
    source_train_loaders,
    source_val_loaders,
    test_loader,
    class_to_idx,
    source_env_names,
) = loader_builder.build()
```

Return values:

- `source_train_loaders`: list of infinite training loaders, one per source environment.
- `source_val_loaders`: list of validation loaders, aligned with `source_env_names`.
- `test_loader`: evaluation loader for the target domain or target domain group.
- `class_to_idx`: shared class-name-to-label mapping.
- `source_env_names`: names of the source environments created by the builder.

Training loaders are infinite, so use a fixed number of steps:

```python
source_iters = [iter(loader) for loader in source_train_loaders]

for step in range(num_steps):
    minibatches = [next(source_iter) for source_iter in source_iters]
    # minibatches is a list of (images, labels), one batch per source env
```

Validation and test loaders are normal finite PyTorch `DataLoader` objects:

```python
for env_name, val_loader in zip(source_env_names, source_val_loaders):
    for images, labels in val_loader:
        pass

for images, labels in test_loader:
    pass
```

## Supported Datasets

### Multi-domain datasets

Use `MultiDomainLoaderBuilder`.

| Dataset | Root path | Domain groups |
| --- | --- | --- |
| DomainNet | `datasets/processed/DomainNet` | `clipart_infograph`, `painting_quickdraw`, `real_sketch` |
| NICOPP | `datasets/processed/NICOPP` | `autumn_rock`, `dim_grass`, `outdoor_water` |

Expected folder layout:

```text
datasets/processed/NICOPP/
  autumn_rock/
    train/
      autumn/
        class_name/
          image.jpg
      rock/
        class_name/
          image.jpg
    test/
      autumn/
      rock/
```

For DomainNet, the real domains inside each group are named after the group parts, for example `clipart` and `infograph` inside `clipart_infograph`.

When `source_domain_groups=None`, all groups except the target group are used as sources. The builder creates one source environment per real domain inside each source group. For example, if the target group is `autumn_rock`, the source environments come from `dim_grass` and `outdoor_water`.

Example:

```python
loader_builder = MultiDomainLoaderBuilder(
    root_path=PROJECT_ROOT / "datasets/processed/NICOPP",
    target_domain_group="autumn_rock",
    source_domain_groups=None,
    train_transform=get_source_transform(),
    eval_transform=get_eval_transform(),
    batch_size=32,
    val_ratio=0.2,
    seed=42,
    num_workers=2,
)

(
    source_train_loaders,
    source_val_loaders,
    test_loader,
    class_to_idx,
    source_env_names,
) = loader_builder.build()
```

To switch to DomainNet, change the root path and target group:

```python
loader_builder = MultiDomainLoaderBuilder(
    root_path=PROJECT_ROOT / "datasets/processed/DomainNet",
    target_domain_group="clipart_infograph",
    train_transform=get_source_transform(),
    eval_transform=get_eval_transform(),
    batch_size=32,
    val_ratio=0.2,
    seed=42,
    num_workers=2,
)
```

### Single-domain datasets

Use `SingleDomainLoaderBuilder`.

| Dataset | Root path | Domains |
| --- | --- | --- |
| OfficeHome | `datasets/processed/OfficeHome` | `Art`, `Clipart`, `Product`, `Real World` |
| PACS | `datasets/processed/PACS` | `art_painting`, `cartoon`, `photo`, `sketch` |
| TerraIncognita | `datasets/processed/TerraIncognita` | `L38`, `L42`, `L46`, `L100` |

Expected folder layout:

```text
datasets/processed/OfficeHome/
  Art/
    class_name/
      image.jpg
  Clipart/
    class_name/
      image.jpg
```

When `source_domains=None`, all domains except the target domain are used as sources.

Example:

```python
loader_builder = SingleDomainLoaderBuilder(
    root_path=PROJECT_ROOT / "datasets/processed/OfficeHome",
    target_domain="Art",
    source_domains=None,
    train_transform=get_source_transform(),
    eval_transform=get_eval_transform(),
    batch_size=32,
    val_ratio=0.2,
    seed=42,
    num_workers=2,
)

(
    source_train_loaders,
    source_val_loaders,
    test_loader,
    class_to_idx,
    source_env_names,
) = loader_builder.build()
```

To use PACS or TerraIncognita, change only `root_path` and `target_domain`:

```python
loader_builder = SingleDomainLoaderBuilder(
    root_path=PROJECT_ROOT / "datasets/processed/PACS",
    target_domain="photo",
    train_transform=get_source_transform(),
    eval_transform=get_eval_transform(),
    batch_size=32,
    val_ratio=0.2,
    seed=42,
    num_workers=2,
)
```

```python
loader_builder = SingleDomainLoaderBuilder(
    root_path=PROJECT_ROOT / "datasets/processed/TerraIncognita",
    target_domain="L38",
    train_transform=get_source_transform(),
    eval_transform=get_eval_transform(),
    batch_size=32,
    val_ratio=0.2,
    seed=42,
    num_workers=2,
)
```

Note: pass the exact folder name that exists under `datasets/processed/TerraIncognita`.

### VLCS

VLCS is special because each domain contains split folders. Use `SplitDomainLoaderBuilder`, not `SingleDomainLoaderBuilder`.

| Dataset | Root path | Domains |
| --- | --- | --- |
| VLCS | `datasets/processed/VLCS` | `CALTECH`, `LABELME`, `PASCAL`, `SUN` |

Expected folder layout:

```text
datasets/processed/VLCS/
  PASCAL/
    train/
      class_name/
        image.jpg
    test/
      class_name/
        image.jpg
  SUN/
    train/
    test/
```

When `source_domains=None`, all VLCS domains except the target domain are used as sources.

Example from the notebook:

```python
loader_builder = SplitDomainLoaderBuilder(
    root_path=PROJECT_ROOT / "datasets/processed/VLCS",
    target_domain="PASCAL",
    source_domains=None,
    source_splits=("train", "test"),
    target_splits=("train", "test"),
    train_transform=get_source_transform(),
    eval_transform=get_eval_transform(),
    batch_size=32,
    val_ratio=0.2,
    seed=42,
    num_workers=2,
)

(
    source_train_loaders,
    source_val_loaders,
    test_loader,
    class_to_idx,
    source_env_names,
) = loader_builder.build()
```

If you want the common train-source/test-target setup, use:

```python
source_splits=("train",)
target_splits=("test",)
```

## Common Parameters

| Parameter | Meaning |
| --- | --- |
| `root_path` | Dataset root folder. For multi-domain datasets, this is the folder containing all domain groups. |
| `target_domain_group` | Target group name for `MultiDomainLoaderBuilder`. |
| `target_domain` | Target domain name for `SingleDomainLoaderBuilder` and `SplitDomainLoaderBuilder`. |
| `source_domain_groups` | Optional list of source groups for multi-domain datasets. Defaults to all groups except the target. |
| `source_domains` | Optional list of source domains. Defaults to all domains except the target. |
| `source_splits` | Splits used to build source loaders for split-based datasets. |
| `target_splits` | Splits used to build the target test loader for split-based datasets. |
| `train_transform` | Transform used for source training subsets. |
| `eval_transform` | Transform used for source validation and target test subsets. |
| `batch_size` | Batch size per source environment. Total training batch size is `batch_size * len(source_train_loaders)`. |
| `val_ratio` | Fraction of each source environment held out for validation. |
| `seed` | Random seed used for train/validation splitting. |
| `num_workers` | PyTorch dataloader workers. Use `0` if notebook multiprocessing causes issues on your machine. |

## Quick Reference

```python
MULTI_DOMAIN_GROUPS = {
    "DomainNet": ["clipart_infograph", "painting_quickdraw", "real_sketch"],
    "NICOPP": ["autumn_rock", "dim_grass", "outdoor_water"],
}

SINGLE_DOMAIN_DOMAINS = {
    "OfficeHome": ["Art", "Clipart", "Product", "Real World"],
    "PACS": ["art_painting", "cartoon", "photo", "sketch"],
    "TerraIncognita": ["L38", "L42", "L46", "L100"],
}

VLCS_DOMAINS = ["CALTECH", "LABELME", "PASCAL", "SUN"]
```
