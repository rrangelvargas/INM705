# imports
import torch
import torch.nn as nn
import wandb
import time
import json
import os
import itertools
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset.dataset import SignLanguageDataset
import argparse
import datetime
import matplotlib.pyplot as plt

# returns a timestamp string for printing/logging
def timestamp():
    return datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

# parses command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline",
        choices=["baseline", "attention", "baseline_lr", "baseline_loss_ls", "baseline_loss_arcface", "baseline_loss_combined", "vqvae"],
        help="Model type to train")
    parser.add_argument("--max-words", type=int, default=None, help="Maximum number of words to use from the dataset")
    return parser.parse_args()

# runs one full training or validation epoch
def run_epoch(model, loader, loss_fn, opt, device, is_train=True):
    model.train() if is_train else model.eval()

    total_loss, correct, total, num_batches = 0.0, 0, 0, 0
    total_class_loss, total_extra_loss = 0.0, 0.0

    loop = tqdm(loader, desc="train" if is_train else "val")
    for x, y in loop:
        x, y = x.to(device), y.squeeze().to(device)

        if is_train:
            opt.zero_grad()

        with torch.set_grad_enabled(is_train):
            output = model(x)
            logits, extra_loss = output if isinstance(output, tuple) else (output, 0.0)
            class_loss = loss_fn(logits, y)
            loss = class_loss + extra_loss

            if is_train:
                loss.backward()
                opt.step()

        total_loss += loss.item()
        total_class_loss += class_loss.item()
        if isinstance(extra_loss, torch.Tensor):
            total_extra_loss += extra_loss.item()

        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        num_batches += 1

        loop.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})

    # returns averaged loss, class loss, extra loss, and accuracy
    return (total_loss / num_batches,
            total_class_loss / num_batches,
            total_extra_loss / num_batches,
            100 * correct / total)

# runs training for one configuration from the grid
def train_with_config(config, combinations):
    start_time = time.time()
    trial_id = config['trial_id']

    print(f"\n{timestamp()} === Starting Trial {trial_id}/{len(combinations)} ===")
    print(f"{timestamp()} Config: {config}")

    # initialize wandb
    wandb.init(
        project="sign-language-lstm",
        config=config,
        name=f"grid-{trial_id}",
        reinit=True,
        group=config["model"],
        tags=[config["model"]]
    )

    config = wandb.config

    # load dataset
    print(f"{timestamp()} Loading dataset...")
    dataset = SignLanguageDataset(
        json_path='data/WLASL_v0.3.json',
        videos_dir='data/videos',
        sequence_length=config.sequence_length,
        cache_dir='data/cached_landmarks',
        max_words=config.max_words
    )
    print(f"{timestamp()} Dataset loaded: {len(dataset)} samples | {len(dataset.word2idx)} classes")

    # split into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    # create data loaders
    loader = lambda ds: DataLoader(ds, batch_size=config.batch_size, shuffle=True, drop_last=True)
    train_loader, val_loader = loader(train_set), loader(val_set)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{timestamp()} Using device: {device}")

    # log device to wandb
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    wandb.config.update({"device": device_name})

    # select model
    if config.model == "baseline":
        from models.model_baseline import SignLanguageModel as SelectedModel
    elif config.model == "baseline_lr":
        from models.model_baseline_lr import SignLanguageModel as SelectedModel
    elif config.model in ["baseline_loss_ls", "baseline_loss_arcface", "baseline_loss_combined"]:
        from models.model_baseline_loss import SignLanguageModel as SelectedModel
    elif config.model == "attention":
        from models.model_attention import SignLanguageAttentionModel as SelectedModel
    else:
        raise ValueError(f"Unknown model type: {config.model}")

    # initialize model
    model = SelectedModel(len(dataset.word2idx)).to(device)
    wandb.watch(model, log="gradients", log_freq=100)

    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn_name = "cross_entropy"

    # select loss function
    if config.model == "baseline_loss_ls":
        from models.model_baseline_loss import get_loss
        loss_fn = get_loss("label_smoothing")
        loss_fn_name = "label_smoothing"
        wandb.config.update({
            "loss_function": "label_smoothing",
            "loss_params": {"reduction": "mean"}
        })
    elif config.model == "baseline_loss_arcface":
        from models.model_baseline_loss import get_loss
        loss_fn = get_loss("arcface", margin=0.5, scale=30.0)
        loss_fn_name = "arcface"
        wandb.config.update({
            "loss_function": "arcface",
            "loss_params": {"margin": 0.5, "scale": 30.0, "reduction": "mean"}
        })
    else:
        loss_fn = nn.CrossEntropyLoss()
        wandb.config.update({
            "loss_function": "cross_entropy",
            "loss_params": {"reduction": "mean"}
        })

    # initialize scheduler if needed
    if config.model == "baseline_lr":
        from models.model_baseline_lr import WarmupCosineScheduler
        scheduler = WarmupCosineScheduler(
            optimizer=opt,
            warmup_epochs=5,
            max_epochs=30,
            min_lr=1e-6
        )
        wandb.config.update({
            "scheduler": "warmup_cosine",
            "scheduler_params": {
                "warmup_epochs": 5,
                "max_epochs": config.epochs,
                "min_lr": 1e-6
            }
        })

    print(f"{timestamp()} Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    best_val_acc = 0.0

    for epoch in range(config.epochs):
        print(f"\n{timestamp()} [Epoch {epoch + 1}/{config.epochs}]")

        # run training and validation
        train_loss, train_class, train_extra, train_acc = run_epoch(model, train_loader, loss_fn, opt, device, True)
        val_loss, val_class, val_extra, val_acc = run_epoch(model, val_loader, loss_fn, opt, device, False)

        # step scheduler if used
        if config.model == "baseline_lr":
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            wandb.log({"learning_rate": current_lr, "epoch": epoch + 1})

        print(f"{timestamp()} Train acc: {train_acc:.2f}% | Val acc: {val_acc:.2f}%")
        print(f"{timestamp()} Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

        # log metrics
        log_data = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_class_loss": train_class,
            "train_extra_loss": train_extra,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_class_loss": val_class,
            "val_extra_loss": val_extra,
            "val_acc": val_acc
        }

        if "baseline_loss" in config.model:
            log_data.update({"loss_type": loss_fn_name, "loss_value": train_class})
        elif config.model == "baseline_lr":
            log_data.update({"current_lr": current_lr, "scheduler_epoch": scheduler.current_epoch})

        wandb.log(log_data)

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = f"models/{config.model}_trial{trial_id}_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"{timestamp()} [SAVE] New best model to {save_path} (val acc: {val_acc:.2f}%)")

    print(f"{timestamp()} Trial {trial_id} complete. Best val acc: {best_val_acc:.2f}%\n")

    # log trial results
    run_time_secs = round(time.time() - start_time, 2)
    os.makedirs("results", exist_ok=True)

    try:
        with open(f"results/trial_{trial_id:03d}_{config.model}_log.json", "w") as f:
            json.dump({
                "timestamp": timestamp(),
                "run_time_secs": run_time_secs,
                "val_acc": best_val_acc,
                "config": dict(config),
                "model_specific": {
                    "loss_function": config.get("loss_function", "cross_entropy"),
                    "loss_params": config.get("loss_params", {}),
                    "scheduler": config.get("scheduler", None),
                    "scheduler_params": config.get("scheduler_params", {})
                }
            }, f, indent=2)
    except Exception as e:
        print(f"{timestamp()} [WARN] Failed to write result JSON: {e}")

    wandb.log({
        "final_val_acc": best_val_acc,
        "run_time_secs": run_time_secs,
        "final_epoch": config.epochs,
        "final_model": config.model,
        "model_specific": {
            "loss_function": config.get("loss_function", "cross_entropy"),
            "loss_params": config.get("loss_params", {}),
            "scheduler": config.get("scheduler", None),
            "scheduler_params": config.get("scheduler_params", {})
        }
    })

    wandb.finish()
    time.sleep(2)

    # append result to grid log
    with open("grid_search_log.txt", "a") as f:
        f.write(f"{timestamp()} Trial {trial_id} | {dict(config)} | Val Acc: {best_val_acc:.2f}\n")

    return best_val_acc

# grid search main loop
def main():
    args = parse_args()

    # define hyperparameter grid
    grid = {
        "batch_size": [8, 16],
        "lr": [1e-4, 3e-4],
        "sequence_length": [40],
        "num_layers": [1],
        "dropout": [0.0, 0.5],
        "max_words": [args.max_words] if args.max_words is not None else [None]
    }

    # adjust grid for attention models
    if args.model == "attention":
        grid["hidden_size"] = [512]
        grid["attention_type"] = ["linear", "additive"]
    else:
        grid["hidden_size"] = [256, 512]

    # generate all trial configs
    param_keys = list(grid.keys())
    combinations = list(itertools.product(*grid.values()))

    for trial_id, values in enumerate(combinations):
        try:
            print(f"{timestamp()} Grid Progress: Trial {trial_id + 1}/{len(combinations)}")

            config = dict(zip(param_keys, values))
            config.update({
                "model": args.model,
                "dataset": "WLASL",
                "epochs": 20,
                "trial_id": trial_id
            })

            train_with_config(config, combinations)

        except Exception as e:
            print(f"{timestamp()} [ERROR] Trial {trial_id} failed: {e}")
            with open(f"grid_search_log_{config['model']}.txt", "a") as f:
                f.write(f"{timestamp()} Trial {trial_id} FAILED | {dict(config)} | Error: {str(e)}\n")

    print(f"{timestamp()} Grid search complete. Logs saved to grid_search_log.txt")

if __name__ == "__main__":
    main()
