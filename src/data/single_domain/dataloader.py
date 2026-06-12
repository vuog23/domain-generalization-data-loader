import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset

from src.data.single_domain.dataset import SingleDomainDataset


class InfiniteDataLoader:
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


def build_single_domain_class_to_idx(root_path):
    root_path = Path(root_path)
    class_names = set()

    for domain_path in root_path.iterdir():
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


def make_single_domain_loaders(
    root_path,
    target_domain,
    source_domains=None,
    train_transform=None,
    eval_transform=None,
    batch_size=32,
    val_ratio=0.2,
    seed=42,
    num_workers=2,
):

    root_path = Path(root_path)

    all_domains = sorted([
        p.name for p in root_path.iterdir()
        if p.is_dir()
    ])

    if target_domain not in all_domains:
        raise ValueError(
            f"Target domain '{target_domain}' not found. "
            f"Available domains: {all_domains}"
        )

    if source_domains is None:
        source_domains = [
            domain for domain in all_domains
            if domain != target_domain
        ]

    class_to_idx = build_single_domain_class_to_idx(root_path)

    source_train_loaders = []
    source_val_loaders = []
    source_env_names = []

    pin_memory = torch.cuda.is_available()

    for env_id, source_domain in enumerate(source_domains):
        train_full = SingleDomainDataset(
            root_path=root_path,
            domains=[source_domain],
            class_to_idx=class_to_idx,
            transform=train_transform,
        )

        val_full = SingleDomainDataset(
            root_path=root_path,
            domains=[source_domain],
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

        source_train_loaders.append(train_loader)
        source_val_loaders.append(val_loader)
        source_env_names.append(source_domain)

        print(
            f"Source env {env_id}: {source_domain} | "
            f"train={len(train_subset)}, val={len(val_subset)}, "
            f"batch_size={batch_size}"
        )

    test_dataset = SingleDomainDataset(
        root_path=root_path,
        domains=[target_domain],
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

    print("\nFinal single-domain DomainBed-style loaders:")
    print(f"Number of source envs: {len(source_train_loaders)}")
    print(f"Batch size per env:    {batch_size}")
    print(f"Total train batch:     {batch_size * len(source_train_loaders)}")
    print(f"Target test size:      {len(test_dataset)}")
    print(f"Num classes:           {len(class_to_idx)}")
    print(f"Target domain:         {target_domain}")
    print(f"Source envs:           {source_env_names}")

    return (
        source_train_loaders,
        source_val_loaders,
        test_loader,
        class_to_idx,
        source_env_names,
    )