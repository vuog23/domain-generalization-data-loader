# DomainBed-Style Training

This folder contains a DomainBed-v2-style training stack adapted for this project.
The algorithm/network code follows `h-yu16/DomainBed-v2`, while dataset loading is
delegated to the existing builders in `src/data/dataloader.py`.

## Files

| File                      | Purpose                                                                       |
| ------------------------- | ----------------------------------------------------------------------------- |
| `config.yaml`           | Central experiment, dataset, hyperparameter, and Bayesian tuning config.      |
| `train.py`              | Main YAML-driven trainer and Optuna/TPE tuning entry point.                   |
| `domainbed/`            | DomainBed-v2-style algorithms, networks, optimizer helpers, and SWAD helpers. |
| `domainbed/datasets.py` | Thin bridge into`src.data.dataloader`; no duplicate dataloader logic.       |

## Install

From the project root:

```powershell
pip install pyyaml optuna torch torchvision numpy pillow tqdm
```

`optuna` is only required for Bayesian tuning. `Fishr` additionally requires
`backpack-for-pytorch`.

## Train

Run the default PACS ERM experiment:

```powershell
python domainbed_style/train.py --config domainbed_style/config.yaml
```

Common overrides:

```powershell
python domainbed_style/train.py --dataset PACS --target photo --algorithm CORAL --steps 5000
```

Outputs are written under:

```text
domainbed_style/outputs/<run_name>/
domainbed_style/logs/<run_name>/
```

The main artifacts are `metrics.json`, `hparams.json`, `resolved_config.yaml`,
and `model_best.pt` when `training.save_model_best` is true.

## Change Dataset

Edit the active `dataset` block in `config.yaml`. The supported local layouts are:

| Dataset            | Builder           | Target key              |
| ------------------ | ----------------- | ----------------------- |
| `DomainNet`      | `multi_domain`  | `target_domain_group` |
| `NICOPP`         | `multi_domain`  | `target_domain_group` |
| `OfficeHome`     | `single_domain` | `target_domain`       |
| `PACS`           | `single_domain` | `target_domain`       |
| `TerraIncognita` | `single_domain` | `target_domain`       |
| `VLCS`           | `split_domain`  | `target_domain`       |

For example, OfficeHome Art:

```yaml
dataset:
  name: OfficeHome
  builder: single_domain
  root_path: datasets/processed/OfficeHome
  target_domain: Art
  source_domains: null
  val_ratio: 0.2
  num_workers: 2
```

`source_domains: null` or `source_domain_groups: null` means "use every source
domain except the target", matching the behavior of the existing loader builders.

## Bayesian Tuning

Set this in `config.yaml`:

```yaml
bayesian_tuning:
  enabled: true
```

Then run:

```powershell
python domainbed_style/train.py --mode tune --trials 20
```

The tuning path uses Optuna's TPE sampler. It optimizes source validation
accuracy and stores the best search summary in:

```text
domainbed_style/outputs/tuning_summaries/
```

Tune a specific algorithm by changing `training.algorithm` and editing
`bayesian_tuning.search_space`. The `algorithm_search_spaces` block in
`config.yaml` contains ready-to-copy ranges for CORAL, MMD, IRM, GroupDRO,
Mixup, Fish, SagNet, RSC, and Fishr.

## Pretraining

`training.pretrain` supports the DomainBed-v2 values:

```yaml
pretrain: Supervised
```

Other supported values include `None`, `MoCo`, `MoCo-v2`, `SimCLR`, `SimCLR-v2`,
and `MoCo-v3` depending on `training.arch`.

For MoCo/SimCLR checkpoints, set:

```yaml
hparams:
  common:
    pretrained_weights_dir: C:/Users/<you>/Pretrained_Weights
```

When this is null, the copied DomainBed-v2 default of `~/Pretrained_Weights` is
used.

## Notes

Training loaders are infinite, one per source environment, exactly as returned by
`MultiDomainLoaderBuilder`, `SingleDomainLoaderBuilder`, or
`SplitDomainLoaderBuilder`. Validation and target test loaders are finite PyTorch
loaders from the same builder.
