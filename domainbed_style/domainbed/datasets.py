"""Local dataset bridge for the DomainBed-style runner.

This module intentionally delegates to the project's existing dataloader
builders in ``src.data``. It does not reimplement sample discovery, splitting,
or transforms.
"""

from copy import deepcopy
from pathlib import Path

from src.data.dataloader import (
    MultiDomainLoaderBuilder,
    SingleDomainLoaderBuilder,
    SplitDomainLoaderBuilder,
)
from src.data.transforms import get_eval_transform, get_source_transform


BUILDER_NAMES = {
    "multi_domain": MultiDomainLoaderBuilder,
    "single_domain": SingleDomainLoaderBuilder,
    "split_domain": SplitDomainLoaderBuilder,
}


DATASET_PRESETS = {
    "DomainNet": {
        "builder": "multi_domain",
        "root_path": "datasets/processed/DomainNet",
        "target_domain_group": "clipart_infograph",
        "source_domain_groups": None,
        "source_splits": ["train", "test"],
        "target_splits": ["train", "test"],
    },
    "NICOPP": {
        "builder": "multi_domain",
        "root_path": "datasets/processed/NICOPP",
        "target_domain_group": "autumn_rock",
        "source_domain_groups": None,
        "source_splits": ["train", "test"],
        "target_splits": ["train", "test"],
    },
    "OfficeHome": {
        "builder": "single_domain",
        "root_path": "datasets/processed/OfficeHome",
        "target_domain": "Art",
        "source_domains": None,
    },
    "PACS": {
        "builder": "single_domain",
        "root_path": "datasets/processed/PACS",
        "target_domain": "photo",
        "source_domains": None,
    },
    "TerraIncognita": {
        "builder": "single_domain",
        "root_path": "datasets/processed/TerraIncognita",
        "target_domain": "L38",
        "source_domains": None,
    },
    "VLCS": {
        "builder": "split_domain",
        "root_path": "datasets/processed/VLCS",
        "target_domain": "PASCAL",
        "source_domains": None,
        "source_splits": ["train"],
        "target_splits": ["test"],
    },
}


def _as_tuple(value):
    if value is None:
        return None
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _resolve_path(path_value, project_root):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(project_root) / path


def resolve_dataset_config(dataset_config):
    """Merge a config block with the built-in dataset preset of the same name."""

    config = deepcopy(dataset_config)
    name = config.get("name")
    preset = deepcopy(DATASET_PRESETS.get(name, {}))
    preset.update({k: v for k, v in config.items() if v is not None})
    return preset


def build_loaders(dataset_config, batch_size, seed, num_workers, project_root):
    """Build DomainBed-style loaders using the exact project dataloader classes."""

    config = resolve_dataset_config(dataset_config)
    builder_name = config.get("builder")
    root_path = _resolve_path(config["root_path"], project_root)

    if builder_name not in BUILDER_NAMES:
        raise ValueError(
            f"Unknown dataset builder '{builder_name}'. "
            f"Expected one of {sorted(BUILDER_NAMES)}."
        )

    common = {
        "root_path": root_path,
        "train_transform": get_source_transform(),
        "eval_transform": get_eval_transform(),
        "batch_size": int(batch_size),
        "val_ratio": float(config.get("val_ratio", 0.2)),
        "seed": int(seed),
        "num_workers": int(num_workers),
    }

    if builder_name == "multi_domain":
        builder = MultiDomainLoaderBuilder(
            target_domain_group=config["target_domain_group"],
            source_domain_groups=config.get("source_domain_groups"),
            source_splits=_as_tuple(config.get("source_splits", ("train", "test"))),
            target_splits=_as_tuple(config.get("target_splits", ("train", "test"))),
            **common,
        )
    elif builder_name == "single_domain":
        builder = SingleDomainLoaderBuilder(
            target_domain=config["target_domain"],
            source_domains=config.get("source_domains"),
            **common,
        )
    else:
        builder = SplitDomainLoaderBuilder(
            target_domain=config["target_domain"],
            source_domains=config.get("source_domains"),
            source_splits=_as_tuple(config.get("source_splits", ("train",))),
            target_splits=_as_tuple(config.get("target_splits", ("test",))),
            **common,
        )

    return builder.build()
