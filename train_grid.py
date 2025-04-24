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

# os.environ["https_proxy"] = "http_proxy=http://hpc-proxy00.city.ac.uk:3128/"

def timestamp():
    return datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline", 
                       choices=["baseline", "vqvae", "attention", "baseline_lr", "baseline_loss_ls", "baseline_loss_arcface"])
    parser.add_argument("--max-words", type=int, default=None, help="Maximum number of words to use from the dataset")
    return parser.parse_args()

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

    return (total_loss / num_batches,
            total_class_loss / num_batches,
            total_extra_loss / num_batches,
            100 * correct / total)

def train_with_config(config, combinations):
    start_time = time.time()
    trial_id = config['trial_id']
    print(f"\n{timestamp()} === Starting Trial {trial_id}/{len(combinations)} ===")
    print(f"{timestamp()} Config: {config}")

    wandb.init(
        project="sign-language-lstm",
        config=config,
        name=f"grid-{trial_id}",
        reinit=True,
        group=config["model"],
        tags=[config["model"]]
    )

    config = wandb.config

    print(f"{timestamp()} Loading dataset...")
    dataset = SignLanguageDataset(
        json_path='data/WLASL_v0.3.json',
        videos_dir='data/videos',
        sequence_length=config.sequence_length,
        cache_dir='data/cached_landmarks',
        max_words=config.max_words
    )
    print(f"{timestamp()} Dataset loaded: {len(dataset)} samples | {len(dataset.word2idx)} classes")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    loader = lambda ds: DataLoader(ds, batch_size=config.batch_size, shuffle=True, drop_last=True)
    train_loader, val_loader = loader(train_set), loader(val_set)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{timestamp()} Using device: {device}")

    # Log device info to wandb and print
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    wandb.config.update({"device": device_name})
    print(f"{timestamp()} Logged device to W&B: {device_name}")

    if config.model == "baseline":
        from models.model_baseline import SignLanguageModel as SelectedModel
        model = SelectedModel(
            num_classes=len(dataset.word2idx),
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout
        )
    elif config.model == "baseline_lr":
        from models.model_baseline_lr import SignLanguageModel as SelectedModel
        model = SelectedModel(
            num_classes=len(dataset.word2idx),
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout
        )
    elif config.model == "baseline_loss":
        from models.model_baseline_loss import SignLanguageModel as SelectedModel
        model = SelectedModel(
            num_classes=len(dataset.word2idx),
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout
        )
    elif config.model == "vqvae":
        from models.model_vqvae import SignLanguageVQVAEModel as SelectedModel
        model = SelectedModel(len(dataset.word2idx))
    elif config.model == "attention":
        from models.model_attention import SignLanguageAttentionModel as SelectedModel
        model = SelectedModel(
            num_classes=len(dataset.word2idx),
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            attention_type=config.attention_type
        )
    else:
        raise ValueError(f"Unknown model: {config.model}")

    model = model.to(device)
    wandb.watch(model, log="gradients", log_freq=100)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)

    loss_fn_name = "cross_entropy"

    # Select appropriate loss function
    if config.model == "baseline_loss_ls":
        from models.model_baseline_loss import get_loss
        loss_fn = get_loss("label_smoothing")
        loss_fn_name = "label_smoothing"
        # Log loss function details
        wandb.config.update({
            "loss_function": "label_smoothing",
            "loss_params": {"reduction": "mean"}
        })
    elif config.model == "baseline_loss_arcface":
        from models.model_baseline_loss import get_loss
        loss_fn = get_loss("arcface", margin=0.5, scale=30.0)
        loss_fn_name = "arcface"
        # Log loss function details
        wandb.config.update({
            "loss_function": "arcface",
            "loss_params": {
                "margin": 0.5,
                "scale": 30.0,
                "reduction": "mean"
            }
        })
    else:
        loss_fn = nn.CrossEntropyLoss()
        wandb.config.update({
            "loss_function": "cross_entropy",
            "loss_params": {"reduction": "mean"}
        })

    # Initialize learning rate scheduler if using baseline_lr
    if config.model == "baseline_lr":
        from models.model_baseline_lr import WarmupCosineScheduler
        scheduler = WarmupCosineScheduler(
            optimizer=opt,
            warmup_epochs=5,
            max_epochs=30,
            min_lr=1e-6
        )
        # Log scheduler details
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
        train_loss, train_class, train_extra, train_acc = run_epoch(model, train_loader, loss_fn, opt, device, True)
        val_loss, val_class, val_extra, val_acc = run_epoch(model, val_loader, loss_fn, opt, device, False)

        # Step the learning rate scheduler if using baseline_lr
        if config.model == "baseline_lr":
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            wandb.log({
                "learning_rate": current_lr,
                "epoch": epoch + 1
            })

        print(f"{timestamp()} Train acc: {train_acc:.2f}% | Val acc: {val_acc:.2f}%")
        print(f"{timestamp()} Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
        print(f"{timestamp()} Logging to W&B: Epoch {epoch + 1} | Val Acc: {val_acc:.2f}")

        # Enhanced logging for different models
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

        # Add model-specific metrics
        if "baseline_loss" in config.model:
            log_data.update({
                "loss_type": loss_fn_name,
                "loss_value": train_class
            })
        elif config.model == "baseline_lr":
            log_data.update({
                "current_lr": current_lr,
                "scheduler_epoch": scheduler.current_epoch
            })

        wandb.log(log_data)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = f"models/{config.model}_trial{trial_id}_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"{timestamp()} [SAVE] New best model to {save_path} (val acc: {val_acc:.2f}%)")

    print(f"{timestamp()} Trial {trial_id} complete. Best val acc: {best_val_acc:.2f}%\n")

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

    with open("grid_search_log.txt", "a") as f:
        f.write(f"{timestamp()} Trial {trial_id} | {dict(config)} | Val Acc: {best_val_acc:.2f}\n")

    return best_val_acc

def main():
    args = parse_args()

    # # Define grid search ranges
    grid = {
        "batch_size": [8, 16],
        "lr": [1e-4, 3e-4],
        "sequence_length": [40],
        "num_layers": [1],
        "dropout": [0.0, 0.5],
        "max_words": [args.max_words] if args.max_words is not None else [None]
    }

    # Modify grid based on model
    if args.model == "attention":
        grid["hidden_size"] = [512]  # Always use 512 for attention
        grid["attention_type"] = ["dot", "additive"]
    else:
        grid["hidden_size"] = [256, 512]  # More options for other models


    # Create all trial combinations
    param_keys = list(grid.keys())
    combinations = list(itertools.product(*grid.values()))

    for trial_id, values in enumerate(combinations):
        try:
            print(f"{timestamp()} Grid Progress: Trial {trial_id + 1}/{len(combinations)}")

            # Build config
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
