import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset

from src.data.nicopp_dataset import NICOPPDataset


class InfiniteDataLoader:
    """
    DomainBed-style infinite loader.
    Each source domain has its own InfiniteDataLoader.
    """
    def __init__(
        self,
        dataset,
        batch_size,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    ):
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=True if num_workers > 0 else False,
        )

    def __iter__(self):
        while True:
            for batch in self.loader:
                yield batch

    def __len__(self):
        return len(self.loader)


def build_class_to_idx(root_path, splits=("train", "test")):
    root_path = Path(root_path)
    class_names = set()

    for group_path in root_path.iterdir():
        if not group_path.is_dir():
            continue

        for split in splits:
            split_path = group_path / split

            if not split_path.exists():
                continue

            for domain_path in split_path.iterdir():
                if not domain_path.is_dir():
                    continue

                for category_path in domain_path.iterdir():
                    if category_path.is_dir():
                        class_names.add(category_path.name)

    if len(class_names) == 0:
        raise ValueError(f"No class folders found inside {root_path}")

    return {
        class_name: idx
        for idx, class_name in enumerate(sorted(class_names))
    }


def get_real_domains(root_path, domain_group, splits=("train", "test")):
    root_path = Path(root_path)
    real_domains = set()

    for split in splits:
        split_path = root_path / domain_group / split

        if not split_path.exists():
            continue

        for domain_path in split_path.iterdir():
            if domain_path.is_dir():
                real_domains.add(domain_path.name)

    return sorted(real_domains)


def split_indices(n, val_ratio=0.2, seed=42):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()

    val_size = int(n * val_ratio)

    if n > 1:
        val_size = max(1, val_size)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    if len(train_indices) == 0:
        raise ValueError("Train split is empty. Reduce val_ratio.")

    return train_indices, val_indices


def make_domainbed_loaders(
    root_path,
    target_domain_group,
    source_domain_groups=None,
    source_splits=("train", "test"),
    target_splits=("test",),
    train_transform=None,
    eval_transform=None,
    batch_size=32,
    val_ratio=0.2,
    seed=42,
    num_workers=2,
):
    """
    DomainBed-style loader builder.

    Returns:
        source_train_loaders: list of infinite loaders, one per source real domain
        source_val_loaders: list of val loaders, one per source real domain
        test_loader: target test loader
        class_to_idx: label mapping
        source_env_names: names of source environments

    Important:
        batch_size=32 means 32 images per source domain.

        If you have 4 source real domains:
            total train batch = 32 * 4 = 128
    """
    root_path = Path(root_path)

    all_domain_groups = sorted([
        p.name for p in root_path.iterdir()
        if p.is_dir()
    ])

    if target_domain_group not in all_domain_groups:
        raise ValueError(
            f"Target domain group '{target_domain_group}' not found. "
            f"Available groups: {all_domain_groups}"
        )

    if source_domain_groups is None:
        source_domain_groups = [
            group for group in all_domain_groups
            if group != target_domain_group
        ]

    class_to_idx = build_class_to_idx(
        root_path=root_path,
        splits=("train", "test"),
    )

    source_train_loaders = []
    source_val_loaders = []
    source_env_names = []

    pin_memory = torch.cuda.is_available()
    env_id = 0

    for domain_group in source_domain_groups:
        real_domains = get_real_domains(
            root_path=root_path,
            domain_group=domain_group,
            splits=source_splits,
        )

        for real_domain in real_domains:
            train_full = NICOPPDataset(
                root_path=root_path,
                domain_groups=[domain_group],
                splits=source_splits,
                real_domains=[real_domain],
                class_to_idx=class_to_idx,
                transform=train_transform,
            )

            val_full = NICOPPDataset(
                root_path=root_path,
                domain_groups=[domain_group],
                splits=source_splits,
                real_domains=[real_domain],
                class_to_idx=class_to_idx,
                transform=eval_transform,
            )

            train_indices, val_indices = split_indices(
                n=len(train_full),
                val_ratio=val_ratio,
                seed=seed + env_id,
            )

            train_subset = Subset(train_full, train_indices)
            val_subset = Subset(val_full, val_indices)

            train_loader = InfiniteDataLoader(
                dataset=train_subset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=True,
            )

            val_loader = DataLoader(
                val_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
                persistent_workers=True if num_workers > 0 else False,
            )

            env_name = f"{domain_group}/{real_domain}"

            source_train_loaders.append(train_loader)
            source_val_loaders.append(val_loader)
            source_env_names.append(env_name)

            print(
                f"Source env {env_id}: {env_name} | "
                f"train={len(train_subset)}, val={len(val_subset)}, "
                f"batch_size={batch_size}"
            )

            env_id += 1

    if len(source_train_loaders) == 0:
        raise ValueError("No source train loaders were created.")

    test_dataset = NICOPPDataset(
        root_path=root_path,
        domain_groups=[target_domain_group],
        splits=target_splits,
        real_domains=None,
        class_to_idx=class_to_idx,
        transform=eval_transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
    )

    print("\nFinal DomainBed-style loaders:")
    print(f"Number of source envs: {len(source_train_loaders)}")
    print(f"Batch size per env:    {batch_size}")
    print(f"Total train batch:     {batch_size * len(source_train_loaders)}")
    print(f"Target test size:      {len(test_dataset)}")
    print(f"Num classes:           {len(class_to_idx)}")
    print(f"Target group:          {target_domain_group}")
    print(f"Source envs:           {source_env_names}")

    return (
        source_train_loaders,
        source_val_loaders,
        test_loader,
        class_to_idx,
        source_env_names,
    )