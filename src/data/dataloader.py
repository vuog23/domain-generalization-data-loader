import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from src.data.dataset import MultiDomainDataset, SingleDomainDataset, SplitDomainDataset

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


class MultiDomainLoaderBuilder:
    def __init__(
        self,
        root_path,
        target_domain_group,
        source_domain_groups=None,
        source_splits=("train", "test"),
        target_splits=("train", "test"),
        train_transform=None,
        eval_transform=None,
        batch_size=32,
        val_ratio=0.2,
        seed=42,
        num_workers=2,
    ):
        self.root_path = Path(root_path)
        self.target_domain_group = target_domain_group
        self.source_domain_groups = source_domain_groups
        self.source_splits = source_splits
        self.target_splits = target_splits
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.seed = seed
        self.num_workers = num_workers

        self.pin_memory = torch.cuda.is_available()

        self.class_to_idx = None
        self.source_env_names = []

    def build_class_to_idx(self, splits=("train", "test")):
        class_names = set()

        for group_path in self.root_path.iterdir():
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
            raise ValueError(f"No class folders found inside {self.root_path}")

        return {
            class_name: idx
            for idx, class_name in enumerate(sorted(class_names))
        }

    def get_real_domains(self, domain_group, splits=("train", "test")):
        real_domains = set()

        for split in splits:
            split_path = self.root_path / domain_group / split

            if not split_path.exists():
                continue

            for domain_path in split_path.iterdir():
                if domain_path.is_dir():
                    real_domains.add(domain_path.name)

        return sorted(real_domains)

    @staticmethod
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

    def get_all_domain_groups(self):
        return sorted([
            p.name for p in self.root_path.iterdir()
            if p.is_dir()
        ])

    def validate_domain_groups(self):
        all_domain_groups = self.get_all_domain_groups()

        if self.target_domain_group not in all_domain_groups:
            raise ValueError(
                f"Target domain group '{self.target_domain_group}' not found. "
                f"Available groups: {all_domain_groups}"
            )

        if self.source_domain_groups is None:
            self.source_domain_groups = [
                group for group in all_domain_groups
                if group != self.target_domain_group
            ]

        return all_domain_groups

    def build_source_loaders(self):
        source_train_loaders = []
        source_val_loaders = []
        source_env_names = []

        env_id = 0

        for domain_group in self.source_domain_groups:
            real_domains = self.get_real_domains(
                domain_group=domain_group,
                splits=self.source_splits,
            )

            for real_domain in real_domains:
                train_full = MultiDomainDataset(
                    root_path=self.root_path,
                    domain_groups=[domain_group],
                splits=self.source_splits,
                    real_domains=[real_domain],
                    class_to_idx=self.class_to_idx,
                    transform=self.train_transform,
                )

                val_full = MultiDomainDataset(
                    root_path=self.root_path,
                    domain_groups=[domain_group],
                    splits=self.source_splits,
                    real_domains=[real_domain],
                    class_to_idx=self.class_to_idx,
                    transform=self.eval_transform,
                )

                train_indices, val_indices = self.split_indices(
                    n=len(train_full),
                    val_ratio=self.val_ratio,
                    seed=self.seed + env_id,
                )

                train_subset = Subset(train_full, train_indices)
                val_subset = Subset(val_full, val_indices)

                train_loader = InfiniteDataLoader(
                    dataset=train_subset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    drop_last=True,
                )

                val_loader = DataLoader(
                    val_subset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    drop_last=False,
                    persistent_workers=True if self.num_workers > 0 else False,
                )

                env_name = f"{domain_group}/{real_domain}"

                source_train_loaders.append(train_loader)
                source_val_loaders.append(val_loader)
                source_env_names.append(env_name)

                print(
                    f"Source env {env_id}: {env_name} | "
                    f"train={len(train_subset)}, val={len(val_subset)}, "
                    f"batch_size={self.batch_size}"
                )

                env_id += 1

        if len(source_train_loaders) == 0:
            raise ValueError("No source train loaders were created.")

        self.source_env_names = source_env_names

        return source_train_loaders, source_val_loaders

    def build_test_loader(self):
        test_dataset = MultiDomainDataset(
            root_path=self.root_path,
            domain_groups=[self.target_domain_group],
            splits=self.target_splits,
            real_domains=None,
            class_to_idx=self.class_to_idx,
            transform=self.eval_transform,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

        return test_loader, test_dataset

    def summary(self, source_train_loaders, test_dataset):
        print("\nFinal DomainBed-style loaders:")
        print(f"Number of source envs: {len(source_train_loaders)}")
        print(f"Batch size per env:    {self.batch_size}")
        print(f"Total train batch:     {self.batch_size * len(source_train_loaders)}")
        print(f"Target test size:      {len(test_dataset)}")
        print(f"Num classes:           {len(self.class_to_idx)}")
        print(f"Target group:          {self.target_domain_group}")
        print(f"Source envs:           {self.source_env_names}")

    def build(self):
        self.validate_domain_groups()

        self.class_to_idx = self.build_class_to_idx(
            splits=("train", "test")
        )

        source_train_loaders, source_val_loaders = self.build_source_loaders()

        test_loader, test_dataset = self.build_test_loader()

        self.summary(
            source_train_loaders=source_train_loaders,
            test_dataset=test_dataset,
        )

        return (
            source_train_loaders,
            source_val_loaders,
            test_loader,
            self.class_to_idx,
            self.source_env_names,
        )


class SingleDomainLoaderBuilder:
    def __init__(
        self,
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
        self.root_path = Path(root_path)
        self.target_domain = target_domain
        self.source_domains = source_domains
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.seed = seed
        self.num_workers = num_workers

        self.pin_memory = torch.cuda.is_available()

        self.class_to_idx = None
        self.source_env_names = []

    def build_class_to_idx(self):
        class_names = set()

        for domain_path in self.root_path.iterdir():
            if not domain_path.is_dir():
                continue

            for category_path in domain_path.iterdir():
                if category_path.is_dir():
                    class_names.add(category_path.name)

        if len(class_names) == 0:
            raise ValueError(f"No class folders found inside {self.root_path}")

        return {
            class_name: idx
            for idx, class_name in enumerate(sorted(class_names))
        }

    @staticmethod
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

    def get_all_domains(self):
        return sorted([
            p.name for p in self.root_path.iterdir()
            if p.is_dir()
        ])

    def validate_domains(self):
        all_domains = self.get_all_domains()

        if self.target_domain not in all_domains:
            raise ValueError(
                f"Target domain '{self.target_domain}' not found. "
                f"Available domains: {all_domains}"
            )

        if self.source_domains is None:
            self.source_domains = [
                domain for domain in all_domains
                if domain != self.target_domain
            ]

        return all_domains

    def build_source_loaders(self):
        source_train_loaders = []
        source_val_loaders = []
        source_env_names = []

        for env_id, source_domain in enumerate(self.source_domains):
            train_full = SingleDomainDataset(
                root_path=self.root_path,
                domains=[source_domain],
                class_to_idx=self.class_to_idx,
                transform=self.train_transform,
            )

            val_full = SingleDomainDataset(
                root_path=self.root_path,
                domains=[source_domain],
                class_to_idx=self.class_to_idx,
                transform=self.eval_transform,
            )

            train_indices, val_indices = self.split_indices(
                n=len(train_full),
                val_ratio=self.val_ratio,
                seed=self.seed + env_id,
            )

            train_subset = Subset(train_full, train_indices)
            val_subset = Subset(val_full, val_indices)

            train_loader = InfiniteDataLoader(
                dataset=train_subset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=True,
            )

            val_loader = DataLoader(
                val_subset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=False,
                persistent_workers=True if self.num_workers > 0 else False,
            )

            source_train_loaders.append(train_loader)
            source_val_loaders.append(val_loader)
            source_env_names.append(source_domain)

            print(
                f"Source env {env_id}: {source_domain} | "
                f"train={len(train_subset)}, val={len(val_subset)}, "
                f"batch_size={self.batch_size}"
            )

        if len(source_train_loaders) == 0:
            raise ValueError("No source train loaders were created.")

        self.source_env_names = source_env_names

        return source_train_loaders, source_val_loaders

    def build_test_loader(self):
        test_dataset = SingleDomainDataset(
            root_path=self.root_path,
            domains=[self.target_domain],
            class_to_idx=self.class_to_idx,
            transform=self.eval_transform,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

        return test_loader, test_dataset

    def summary(self, source_train_loaders, test_dataset):
        print("\nFinal single-domain DomainBed-style loaders:")
        print(f"Number of source envs: {len(source_train_loaders)}")
        print(f"Batch size per env:    {self.batch_size}")
        print(f"Total train batch:     {self.batch_size * len(source_train_loaders)}")
        print(f"Target test size:      {len(test_dataset)}")
        print(f"Num classes:           {len(self.class_to_idx)}")
        print(f"Target domain:         {self.target_domain}")
        print(f"Source envs:           {self.source_env_names}")

    def build(self):
        self.validate_domains()

        self.class_to_idx = self.build_class_to_idx()

        source_train_loaders, source_val_loaders = self.build_source_loaders()

        test_loader, test_dataset = self.build_test_loader()

        self.summary(
            source_train_loaders=source_train_loaders,
            test_dataset=test_dataset,
        )

        return (
            source_train_loaders,
            source_val_loaders,
            test_loader,
            self.class_to_idx,
            self.source_env_names,
        )


class SplitDomainLoaderBuilder:
    def __init__(
        self,
        root_path,
        target_domain,
        source_domains=None,
        source_splits=("train",),
        target_splits=("test",),
        train_transform=None,
        eval_transform=None,
        batch_size=32,
        val_ratio=0.2,
        seed=42,
        num_workers=2,
    ):
        self.root_path = Path(root_path)
        self.target_domain = target_domain
        self.source_domains = source_domains
        self.source_splits = source_splits
        self.target_splits = target_splits
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.seed = seed
        self.num_workers = num_workers

        self.pin_memory = torch.cuda.is_available()

        self.class_to_idx = None
        self.source_env_names = []

    def build_class_to_idx(self, splits=("train", "test")):
        class_names = set()

        for domain_path in self.root_path.iterdir():
            if not domain_path.is_dir():
                continue

            for split in splits:
                split_path = domain_path / split

                if not split_path.exists():
                    continue

                for class_path in split_path.iterdir():
                    if class_path.is_dir():
                        class_names.add(class_path.name)

        if len(class_names) == 0:
            raise ValueError(f"No class folders found inside {self.root_path}")

        return {
            class_name: idx
            for idx, class_name in enumerate(sorted(class_names))
        }

    @staticmethod
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

    def get_all_domains(self):
        return sorted([
            p.name for p in self.root_path.iterdir()
            if p.is_dir()
        ])

    def validate_domains(self):
        all_domains = self.get_all_domains()

        if self.target_domain not in all_domains:
            raise ValueError(
                f"Target domain '{self.target_domain}' not found. "
                f"Available domains: {all_domains}"
            )

        if self.source_domains is None:
            self.source_domains = [
                domain for domain in all_domains
                if domain != self.target_domain
            ]

        return all_domains

    def build_source_loaders(self):
        source_train_loaders = []
        source_val_loaders = []
        source_env_names = []

        for env_id, source_domain in enumerate(self.source_domains):
            train_full = SplitDomainDataset(
                root_path=self.root_path,
                domains=[source_domain],
                splits=self.source_splits,
                class_to_idx=self.class_to_idx,
                transform=self.train_transform,
            )

            val_full = SplitDomainDataset(
                root_path=self.root_path,
                domains=[source_domain],
                splits=self.source_splits,
                class_to_idx=self.class_to_idx,
                transform=self.eval_transform,
            )

            train_indices, val_indices = self.split_indices(
                n=len(train_full),
                val_ratio=self.val_ratio,
                seed=self.seed + env_id,
            )

            train_subset = Subset(train_full, train_indices)
            val_subset = Subset(val_full, val_indices)

            train_loader = InfiniteDataLoader(
                dataset=train_subset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=True,
            )

            val_loader = DataLoader(
                val_subset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=False,
                persistent_workers=True if self.num_workers > 0 else False,
            )

            source_train_loaders.append(train_loader)
            source_val_loaders.append(val_loader)
            source_env_names.append(source_domain)

            print(
                f"Source env {env_id}: {source_domain} | "
                f"train={len(train_subset)}, val={len(val_subset)}, "
                f"batch_size={self.batch_size}"
            )

        if len(source_train_loaders) == 0:
            raise ValueError("No source train loaders were created.")

        self.source_env_names = source_env_names

        return source_train_loaders, source_val_loaders

    def build_test_loader(self):
        test_dataset = SplitDomainDataset(
            root_path=self.root_path,
            domains=[self.target_domain],
            splits=self.target_splits,
            class_to_idx=self.class_to_idx,
            transform=self.eval_transform,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

        return test_loader, test_dataset

    def summary(self, source_train_loaders, test_dataset):
        print("\nFinal split-domain DomainBed-style loaders:")
        print(f"Number of source envs: {len(source_train_loaders)}")
        print(f"Batch size per env:    {self.batch_size}")
        print(f"Total train batch:     {self.batch_size * len(source_train_loaders)}")
        print(f"Target test size:      {len(test_dataset)}")
        print(f"Num classes:           {len(self.class_to_idx)}")
        print(f"Target domain:         {self.target_domain}")
        print(f"Source envs:           {self.source_env_names}")

    def build(self):
        self.validate_domains()

        self.class_to_idx = self.build_class_to_idx(
            splits=("train", "test")
        )

        source_train_loaders, source_val_loaders = self.build_source_loaders()

        test_loader, test_dataset = self.build_test_loader()

        self.summary(
            source_train_loaders=source_train_loaders,
            test_dataset=test_dataset,
        )

        return (
            source_train_loaders,
            source_val_loaders,
            test_loader,
            self.class_to_idx,
            self.source_env_names,
        )