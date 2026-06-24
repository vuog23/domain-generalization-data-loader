"""YAML-driven DomainBed-v2 style training with local project dataloaders."""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from copy import deepcopy
from math import ceil
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

for path in (PROJECT_ROOT, SCRIPT_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "PyYAML is required for domainbed_style/train.py. "
        "Install it with: pip install pyyaml"
    ) from exc

from domainbed import algorithms, hparams_registry
from domainbed.datasets import build_loaders, resolve_dataset_config
from domainbed.lib import misc, swa_utils
from domainbed.lib.swad import LossValley


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def to_builtin(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, defaultdict):
        return dict(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_builtin(payload), handle, indent=2)


def write_yaml(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(to_builtin(payload), handle, sort_keys=False)


def make_logger(log_dir, run_name):
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"domainbed_style.{run_name}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def get_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def prepare_config(config, cli_args):
    config = deepcopy(config)
    config.setdefault("experiment", {})
    config.setdefault("dataset", {})
    config.setdefault("training", {})
    config.setdefault("hparams", {})
    config.setdefault("bayesian_tuning", {})

    if cli_args.dataset:
        preserved = {
            key: config["dataset"][key]
            for key in ("val_ratio", "num_workers")
            if key in config["dataset"]
        }
        config["dataset"] = {"name": cli_args.dataset, **preserved}
    if cli_args.target:
        config["dataset"]["target_domain"] = cli_args.target
        config["dataset"]["target_domain_group"] = cli_args.target
    if cli_args.algorithm:
        config["training"]["algorithm"] = cli_args.algorithm
    if cli_args.steps is not None:
        config["training"]["steps"] = cli_args.steps
    if cli_args.seed is not None:
        config["experiment"]["seed"] = cli_args.seed
    if cli_args.device:
        config["experiment"]["device"] = cli_args.device
    if cli_args.output_dir:
        config["experiment"]["output_dir"] = cli_args.output_dir
    if cli_args.trials is not None:
        config["bayesian_tuning"]["n_trials"] = cli_args.trials
    if cli_args.mode:
        config["training"]["mode"] = cli_args.mode

    return config


def make_hparams(config, overrides=None):
    training = config["training"]
    dataset_name = config["dataset"].get("name", "Custom")
    algorithm_name = training.get("algorithm", "ERM")

    hparams = hparams_registry.default_hparams(algorithm_name, dataset_name)
    configured_hparams = config.get("hparams", {})
    if "common" in configured_hparams or "algorithms" in configured_hparams:
        common_hparams = configured_hparams.get("common", {})
        algorithm_hparams = configured_hparams.get("algorithms", {}).get(algorithm_name, {})
        hparams.update({k: v for k, v in common_hparams.items() if v is not None})
        hparams.update({k: v for k, v in algorithm_hparams.items() if v is not None})
    else:
        hparams.update({k: v for k, v in configured_hparams.items() if v is not None})
    if overrides:
        hparams.update({k: v for k, v in overrides.items() if v is not None})

    hparams["steps"] = int(training.get("steps", hparams.get("steps", 5000)))
    hparams["pretrain"] = training.get("pretrain", "Supervised")
    hparams["linear_probe"] = bool(training.get("linear_probe", False))
    hparams["arch"] = training.get("arch", "resnet50")
    hparams["optimizer"] = training.get("optimizer", "Adam")
    hparams["scheduler"] = training.get("scheduler", "None")
    hparams["swad"] = bool(training.get("swad", False))
    hparams["batch_size"] = int(hparams.get("batch_size", training.get("batch_size", 32)))

    return hparams


def target_name_from_config(dataset_config):
    resolved = resolve_dataset_config(dataset_config)
    if resolved.get("builder") == "multi_domain":
        return resolved.get("target_domain_group", "target")
    return resolved.get("target_domain", "target")


def make_run_name(config, hparams, trial_number=None):
    dataset_name = config["dataset"].get("name", "Custom")
    target_name = target_name_from_config(config["dataset"])
    training = config["training"]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    parts = [
        dataset_name,
        str(target_name).replace(" ", "_"),
        training.get("algorithm", "ERM"),
        hparams.get("arch", "resnet50"),
        str(hparams.get("pretrain", "Supervised")).replace("/", "-"),
        timestamp,
    ]
    if trial_number is not None:
        parts.insert(0, f"trial_{trial_number:03d}")
    return "_".join(parts)


def evaluate_validation(algorithm, val_loaders, source_env_names, device):
    correct_total = 0
    sample_total = 0
    loss_total = 0.0
    by_env = {}

    for env_name, loader in zip(source_env_names, val_loaders):
        acc, correct, loss, loss_sum, total = misc.accuracy_and_loss(
            algorithm, loader, None, device
        )
        by_env[env_name] = {"accuracy": acc, "loss": loss, "total": total}
        correct_total += correct
        sample_total += total
        loss_total += loss_sum

    overall_acc = correct_total / sample_total if sample_total else 0.0
    overall_loss = loss_total / sample_total if sample_total else float("inf")
    return {
        "by_env": by_env,
        "overall": overall_acc,
        "loss": overall_loss,
        "total": sample_total,
    }


def evaluate_test(algorithm, test_loader, dataset_config, device):
    target_name = target_name_from_config(dataset_config)
    acc, correct, total = misc.accuracy(algorithm, test_loader, None, device)
    return {
        "by_env": {
            target_name: {
                "accuracy": acc,
                "total": total,
            }
        },
        "overall": acc,
        "total": total,
        "correct": correct,
    }


def save_checkpoint(path, algorithm, config, hparams, class_to_idx, source_env_names, metrics):
    payload = {
        "config": to_builtin(config),
        "hparams": to_builtin(hparams),
        "class_to_idx": class_to_idx,
        "source_env_names": source_env_names,
        "metrics": to_builtin(metrics),
        "model_dict": algorithm.state_dict(),
    }
    torch.save(payload, path)


def run_training(config, hparam_overrides=None, trial_number=None, save_model=True):
    experiment = config["experiment"]
    training = config["training"]
    dataset_config = config["dataset"]

    seed = int(experiment.get("seed", 0))
    misc.setup_seed(seed)

    hparams = make_hparams(config, hparam_overrides)
    run_name = make_run_name(config, hparams, trial_number=trial_number)

    output_base = resolve_path(experiment.get("output_dir", "domainbed_style/outputs"))
    log_base = resolve_path(experiment.get("log_dir", "domainbed_style/logs"))
    run_dir = output_base / run_name
    log_dir = log_base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = make_logger(log_dir, run_name)

    write_yaml(run_dir / "resolved_config.yaml", config)
    write_json(run_dir / "hparams.json", hparams)

    device = get_device(experiment.get("device", "auto"))
    if hparams["swad"] and device.type != "cuda":
        logger.warning("SWAD is disabled because the copied SWAD helper expects CUDA.")
        hparams["swad"] = False

    logger.info("Run directory: %s", run_dir)
    logger.info("Device: %s", device)
    logger.info("Algorithm: %s", training.get("algorithm", "ERM"))
    logger.info("Dataset: %s", dataset_config.get("name", "Custom"))
    logger.info("Target: %s", target_name_from_config(dataset_config))

    num_workers = int(dataset_config.get("num_workers", experiment.get("num_workers", 2)))
    (
        source_train_loaders,
        source_val_loaders,
        test_loader,
        class_to_idx,
        source_env_names,
    ) = build_loaders(
        dataset_config=dataset_config,
        batch_size=hparams["batch_size"],
        seed=seed,
        num_workers=num_workers,
        project_root=PROJECT_ROOT,
    )

    steps = int(training.get("steps", hparams["steps"]))
    steps_per_epoch = max(1, min(max(1, len(loader)) for loader in source_train_loaders))
    hparams["epochs"] = max(1, ceil(steps / steps_per_epoch))
    write_json(run_dir / "hparams.json", hparams)

    algorithm_class = algorithms.get_algorithm_class(training.get("algorithm", "ERM"))
    algorithm = algorithm_class(
        input_shape=(3, int(training.get("image_size", 224)), int(training.get("image_size", 224))),
        num_classes=len(class_to_idx),
        num_domains=len(source_train_loaders),
        hparams=hparams,
    )
    algorithm.to(device)

    checkpoint_freq = int(training.get("checkpoint_freq", 300))
    if checkpoint_freq <= 0:
        checkpoint_freq = max(1, steps // 5)
    log_freq = max(1, int(training.get("log_freq", 20)))

    logger.info("Classes: %d", len(class_to_idx))
    logger.info("Source environments: %s", source_env_names)
    logger.info("Steps: %d", steps)
    logger.info("Steps per epoch: %d", steps_per_epoch)
    logger.info("Epochs: %d", hparams["epochs"])
    logger.info("Checkpoint frequency: %d", checkpoint_freq)

    train_iterator = zip(*[iter(loader) for loader in source_train_loaders])
    running_metrics = defaultdict(list)
    records = []
    best_val = {"overall": -1.0, "loss": float("inf")}
    best_test = {"overall": 0.0}
    best_step = 0

    if hparams["swad"]:
        swad_algorithm = swa_utils.AveragedModel(algorithm)
        swad = LossValley(
            hparams["n_converge"],
            hparams["n_tolerance"],
            hparams["tolerance_ratio"],
        )
    else:
        swad_algorithm = None
        swad = None

    for step in range(steps):
        minibatches = [
            (x.to(device), y.to(device, dtype=torch.long))
            for x, y in next(train_iterator)
        ]
        step_metrics = algorithm.update(minibatches)

        for key, value in step_metrics.items():
            if isinstance(value, (float, int, np.floating, np.integer)):
                running_metrics[key].append(float(value))

        if swad_algorithm is not None:
            swad_algorithm.update_parameters(algorithm, step=step + 1)

        if (step + 1) % log_freq == 0:
            metric_text = ", ".join(
                f"{key}: {np.mean(values):.4f}"
                for key, values in sorted(running_metrics.items())
                if values
            )
            logger.info("Step %d/%d, epoch %d, %s", step + 1, steps, (step + 1) // steps_per_epoch, metric_text)
            running_metrics.clear()

        if (step + 1) % steps_per_epoch == 0:
            algorithm.scheduler_step()
            logger.info("Next lr: %.8f", algorithm.get_lr()[0])

        should_evaluate = (step + 1) % checkpoint_freq == 0 or step == steps - 1
        if not should_evaluate:
            continue

        logger.info("Evaluating at step %d", step + 1)
        val_metrics = evaluate_validation(
            algorithm,
            source_val_loaders,
            source_env_names,
            device,
        )
        test_metrics = evaluate_test(algorithm, test_loader, dataset_config, device)

        record = {
            "step": step + 1,
            "val": val_metrics,
            "test": test_metrics,
        }
        records.append(record)
        logger.info("Val overall: %.4f, loss: %.4f", val_metrics["overall"], val_metrics["loss"])
        logger.info("Test overall: %.4f", test_metrics["overall"])

        if swad is not None:
            swad.update_and_evaluate(swad_algorithm, val_metrics["overall"], val_metrics["loss"])
            if swad.dead_valley:
                logger.info("SWAD loss valley ended; stopping early.")
                break
            swad_algorithm = swa_utils.AveragedModel(algorithm)

        if val_metrics["overall"] > best_val["overall"]:
            best_val = deepcopy(val_metrics)
            best_test = deepcopy(test_metrics)
            best_step = step + 1
            logger.info("New best validation accuracy at step %d", best_step)
            if save_model and bool(training.get("save_model_best", True)):
                save_checkpoint(
                    run_dir / "model_best.pt",
                    algorithm,
                    config,
                    hparams,
                    class_to_idx,
                    source_env_names,
                    {"best_val": best_val, "best_test": best_test, "best_step": best_step},
                )

        if save_model and bool(training.get("save_model_every_checkpoint", False)):
            save_checkpoint(
                run_dir / f"model_step_{step + 1}.pt",
                algorithm,
                config,
                hparams,
                class_to_idx,
                source_env_names,
                record,
            )

    if swad is not None:
        logger.info("Evaluating final SWAD model.")
        swad_model = swad.get_final_model()
        best_test = evaluate_test(swad_model, test_loader, dataset_config, device)
        logger.info("SWAD test overall: %.4f", best_test["overall"])

    result = {
        "run_dir": str(run_dir),
        "log_dir": str(log_dir),
        "best_step": best_step,
        "best_val": best_val,
        "best_test": best_test,
        "records": records,
        "hparams": hparams,
        "source_env_names": source_env_names,
        "class_to_idx": class_to_idx,
    }
    write_json(run_dir / "metrics.json", result)
    logger.info("Best val overall: %.4f", best_val["overall"])
    logger.info("Best test overall: %.4f", best_test["overall"])
    return result


def suggest_value(trial, name, spec):
    spec_type = spec.get("type", spec.get("_type"))
    values = spec.get("values", spec.get("choices", spec.get("_value")))

    if spec_type in {"choice", "categorical"}:
        return trial.suggest_categorical(name, values)
    if spec_type in {"float", "uniform"}:
        low = float(spec.get("low", values[0]))
        high = float(spec.get("high", values[1]))
        return trial.suggest_float(name, low, high)
    if spec_type in {"log_float", "loguniform"}:
        low = float(spec.get("low", values[0]))
        high = float(spec.get("high", values[1]))
        return trial.suggest_float(name, low, high, log=True)
    if spec_type in {"int", "randint"}:
        low = int(spec.get("low", values[0]))
        high = int(spec.get("high", values[1]))
        if spec_type == "randint":
            high -= 1
        step = int(spec.get("step", 1))
        return trial.suggest_int(name, low, high, step=step)
    if spec_type == "log_int":
        low = int(spec.get("low", values[0]))
        high = int(spec.get("high", values[1]))
        return trial.suggest_int(name, low, high, log=True)

    raise ValueError(f"Unsupported search-space type for '{name}': {spec_type}")


def run_tuning(config):
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "Optuna is required for Bayesian tuning. Install it with: pip install optuna"
        ) from exc

    tuning = config["bayesian_tuning"]
    search_space = tuning.get("search_space", {})
    if not search_space:
        raise ValueError("bayesian_tuning.search_space is empty.")

    sampler_name = tuning.get("sampler", "tpe").lower()
    seed = int(tuning.get("seed", config["experiment"].get("seed", 0)))
    if sampler_name != "tpe":
        raise ValueError("Only the Optuna TPE sampler is configured for Bayesian tuning.")

    sampler = optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=int(tuning.get("startup_trials", 5)),
        multivariate=bool(tuning.get("multivariate", True)),
    )

    pruner = optuna.pruners.NopPruner()
    if tuning.get("pruner", "none").lower() == "median":
        pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(
        study_name=tuning.get("study_name"),
        storage=tuning.get("storage"),
        load_if_exists=bool(tuning.get("storage")),
        direction=tuning.get("direction", "maximize"),
        sampler=sampler,
        pruner=pruner,
    )

    def objective(trial):
        overrides = {
            name: suggest_value(trial, name, spec)
            for name, spec in search_space.items()
        }
        trial_config = deepcopy(config)
        trial_steps = tuning.get("trial_steps")
        if trial_steps is not None:
            trial_config["training"]["steps"] = int(trial_steps)
        trial_config["training"]["save_model_best"] = bool(tuning.get("save_trial_models", False))
        result = run_training(
            trial_config,
            hparam_overrides=overrides,
            trial_number=trial.number,
            save_model=bool(tuning.get("save_trial_models", False)),
        )
        trial.set_user_attr("run_dir", result["run_dir"])
        trial.set_user_attr("best_step", result["best_step"])
        trial.set_user_attr("best_test_overall", result["best_test"]["overall"])
        return result["best_val"]["overall"]

    study.optimize(objective, n_trials=int(tuning.get("n_trials", 20)))

    output_base = resolve_path(config["experiment"].get("output_dir", "domainbed_style/outputs"))
    summary_dir = output_base / "tuning_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{study.study_name or 'study'}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    summary = {
        "study_name": study.study_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial": study.best_trial.number,
        "trials": [
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": str(trial.state),
                "user_attrs": trial.user_attrs,
            }
            for trial in study.trials
        ],
    }
    write_json(summary_path, summary)
    write_yaml(summary_dir / "best_hparams.yaml", study.best_params)

    if bool(tuning.get("refit_best", False)):
        final_config = deepcopy(config)
        if tuning.get("refit_steps") is not None:
            final_config["training"]["steps"] = int(tuning["refit_steps"])
        run_training(final_config, hparam_overrides=study.best_params, save_model=True)

    print(f"Best trial: {study.best_trial.number}")
    print(f"Best val accuracy: {study.best_value:.4f}")
    print(f"Best params: {json.dumps(study.best_params, indent=2)}")
    print(f"Tuning summary: {summary_path}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Train DomainBed-style models with local dataloaders.")
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--mode", choices=["train", "tune"], default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--algorithm", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--trials", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = prepare_config(read_yaml(args.config), args)
    mode = config["training"].get("mode", "train")
    if config.get("bayesian_tuning", {}).get("enabled", False):
        mode = "tune"

    if mode == "tune":
        run_tuning(config)
    else:
        run_training(config)


if __name__ == "__main__":
    main()
