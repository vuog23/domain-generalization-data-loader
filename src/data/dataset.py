from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class MultiDomainDataset(Dataset):
    def __init__(
        self,
        root_path,
        domain_groups,
        splits=("train", "test"),
        real_domains=None,
        class_to_idx=None,
        transform=None,
        return_metadata=False,
    ):
        self.root_path = Path(root_path)

        self.domain_groups = (
            list(domain_groups)
            if isinstance(domain_groups, (list, tuple, set))
            else [domain_groups]
        )

        self.splits = (
            list(splits)
            if isinstance(splits, (list, tuple, set))
            else [splits]
        )

        self.real_domains = set(real_domains) if real_domains is not None else None
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.return_metadata = return_metadata
        self.samples = []

        if self.class_to_idx is None:
            raise ValueError("Please pass class_to_idx from build_class_to_idx().")

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for group_name in self.domain_groups:
            group_path = self.root_path / group_name

            if not group_path.exists():
                raise FileNotFoundError(f"Domain group not found: {group_path}")

            for split in self.splits:
                split_path = group_path / split

                if not split_path.exists():
                    continue

                for domain_path in split_path.iterdir():
                    if not domain_path.is_dir():
                        continue

                    domain_name = domain_path.name

                    if self.real_domains is not None and domain_name not in self.real_domains:
                        continue

                    for category_path in domain_path.iterdir():
                        if not category_path.is_dir():
                            continue

                        category_name = category_path.name

                        if category_name not in self.class_to_idx:
                            continue

                        label = self.class_to_idx[category_name]

                        for img_path in category_path.rglob("*"):
                            if img_path.is_file() and img_path.suffix.lower() in valid_exts:
                                self.samples.append({
                                    "image_path": img_path,
                                    "label": label,
                                    "category": category_name,
                                    "domain": domain_name,
                                    "group": group_name,
                                    "split": split,
                                })

        if len(self.samples) == 0:
            raise ValueError(
                f"No images found for domain_groups={self.domain_groups}, "
                f"splits={self.splits}, real_domains={self.real_domains}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        label = sample["label"]

        if self.transform is not None:
            image = self.transform(image)

        if self.return_metadata:
            return image, label, {
                "image_path": str(sample["image_path"]),
                "category": sample["category"],
                "domain": sample["domain"],
                "group": sample["group"],
                "split": sample["split"],
            }

        return image, label


class SingleDomainDataset(Dataset):
    def __init__(
        self,
        root_path,
        domains,
        class_to_idx=None,
        transform=None,
        return_metadata=False,
    ):
        self.root_path = Path(root_path)

        self.domains = (
            list(domains)
            if isinstance(domains, (list, tuple, set))
            else [domains]
        )

        self.class_to_idx = class_to_idx
        self.transform = transform
        self.return_metadata = return_metadata
        self.samples = []

        if self.class_to_idx is None:
            raise ValueError("Please pass class_to_idx.")

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for domain_name in self.domains:
            domain_path = self.root_path / domain_name

            if not domain_path.exists():
                raise FileNotFoundError(f"Domain folder not found: {domain_path}")

            for category_path in domain_path.iterdir():
                if not category_path.is_dir():
                    continue

                category_name = category_path.name

                if category_name not in self.class_to_idx:
                    continue

                label = self.class_to_idx[category_name]

                for img_path in category_path.rglob("*"):
                    if img_path.is_file() and img_path.suffix.lower() in valid_exts:
                        self.samples.append({
                            "image_path": img_path,
                            "label": label,
                            "category": category_name,
                            "domain": domain_name,
                        })

        if len(self.samples) == 0:
            raise ValueError(
                f"No images found for domains={self.domains} inside {self.root_path}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        label = sample["label"]

        if self.transform is not None:
            image = self.transform(image)

        if self.return_metadata:
            return image, label, {
                "image_path": str(sample["image_path"]),
                "category": sample["category"],
                "domain": sample["domain"],
            }

        return image, label


class SplitDomainDataset(Dataset):
    def __init__(
        self,
        root_path,
        domains,
        splits=("train", "test"),
        class_to_idx=None,
        transform=None,
        return_metadata=False,
    ):
        self.root_path = Path(root_path)

        self.domains = (
            list(domains)
            if isinstance(domains, (list, tuple, set))
            else [domains]
        )

        self.splits = (
            list(splits)
            if isinstance(splits, (list, tuple, set))
            else [splits]
        )

        self.class_to_idx = class_to_idx
        self.transform = transform
        self.return_metadata = return_metadata
        self.samples = []

        if self.class_to_idx is None:
            raise ValueError("Please pass class_to_idx.")

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for domain_name in self.domains:
            domain_path = self.root_path / domain_name

            if not domain_path.exists():
                raise FileNotFoundError(f"Domain folder not found: {domain_path}")

            for split in self.splits:
                split_path = domain_path / split

                if not split_path.exists():
                    continue

                for class_path in split_path.iterdir():
                    if not class_path.is_dir():
                        continue

                    class_name = class_path.name

                    if class_name not in self.class_to_idx:
                        continue

                    label = self.class_to_idx[class_name]

                    for img_path in class_path.rglob("*"):
                        if img_path.is_file() and img_path.suffix.lower() in valid_exts:
                            self.samples.append({
                                "image_path": img_path,
                                "label": label,
                                "category": class_name,
                                "domain": domain_name,
                                "split": split,
                            })

        if len(self.samples) == 0:
            raise ValueError(
                f"No images found for domains={self.domains}, "
                f"splits={self.splits}, root={self.root_path}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample["image_path"]).convert("RGB")
        label = sample["label"]

        if self.transform is not None:
            image = self.transform(image)

        if self.return_metadata:
            return image, label, {
                "image_path": str(sample["image_path"]),
                "category": sample["category"],
                "domain": sample["domain"],
                "split": sample["split"],
            }

        return image, label