import wandb
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os
import datetime
from visualizer import parse_baseline_log, plot_training_progress
from dataset.dataset import SignLanguageDataset

# os.environ["https_proxy"] = "http_proxy=http://hpc-proxy00.city.ac.uk:3128/"

def timestamp():
    return datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

def parse_args():
    parser = argparse.ArgumentParser(description="Train a sign language model.")
    parser.add_argument("--model", type=str, default="baseline", 
                       choices=["baseline", "attention", "baseline_lr", "baseline_loss_ls", "baseline_loss_arcface"],
                       help="Model type to train")
    parser.add_argument("--max-words", type=int, default=None, help="Maximum number of words to use from the dataset")
    return parser.parse_args()

def run_epoch(model, loader, loss_fn, opt, device, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss, correct, total, num_batches = 0.0, 0, 0, 0
    total_class_loss, total_extra_loss = 0.0, 0.0

    loop = tqdm(loader, desc="training" if is_train else "validation")
    for x, y in loop:
        x, y = x.to(device), y.squeeze().to(device)

        if is_train:
            opt.zero_grad()

        with torch.set_grad_enabled(is_train):
            output = model(x)

            # handle models that return (logits, extra_loss) or just logits
            if isinstance(output, tuple):
                logits, extra_loss = output
            else:
                logits, extra_loss = output, 0.0

            class_loss = loss_fn(logits, y)
            total_batch_loss = class_loss + extra_loss

            if is_train:
                total_batch_loss.backward()
                opt.step()

        total_loss += total_batch_loss.item()
        total_class_loss += class_loss.item()
        if isinstance(extra_loss, torch.Tensor):
            total_extra_loss += extra_loss.item()
        num_batches += 1

        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        acc = 100 * correct / total
        loop.set_postfix({
            'loss': f'{total_batch_loss.item():.4f}',
            'acc': f'{acc:.2f}%'
        })

    avg_loss = total_loss / num_batches
    avg_class_loss = total_class_loss / num_batches
    avg_extra_loss = total_extra_loss / num_batches
    accuracy = 100 * correct / total

    return avg_loss, avg_class_loss, avg_extra_loss, accuracy

def train():
    args = parse_args()

    print("[wandb] Initializing W&B run...")
    wandb.init(
        project="sign-language-lstm",
        config={
            "epochs": 20,
            "batch_size": 16,
            "lr": 1e-3,
            "sequence_length": 30,
            "model": args.model,
            "dataset": "WLASL"
        }
    )
    config = wandb.config

    print(f"[config] Model type selected: {config.model}")
    if config.model == "baseline":
        from models.model_baseline import SignLanguageModel as SelectedModel
    elif config.model == "baseline_lr":
        from models.model_baseline_lr import SignLanguageModel as SelectedModel
    elif config.model == "baseline_loss_ls":
        from models.model_baseline_loss import SignLanguageModel as SelectedModel
    elif config.model == "baseline_loss_arcface":
        from models.model_baseline_loss import SignLanguageModel as SelectedModel
    elif config.model == "attention":
        from models.model_attention import SignLanguageAttentionModel as SelectedModel
    else:
        raise ValueError(f"Unknown model type: {config.model}")

    print("[data] Loading dataset...")
    dataset = SignLanguageDataset(
        json_path='data/WLASL_v0.3.json',
        videos_dir='data/videos',
        sequence_length=config.sequence_length,
        cache_dir='data/cached_landmarks',
        max_words=args.max_words
    )
    print(f"[data] Loaded {len(dataset)} samples with {len(dataset.word2idx)} classes.")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    loader = lambda ds: DataLoader(ds, batch_size=config.batch_size, shuffle=True, num_workers=0, drop_last=True)
    train_loader, val_loader = loader(train_set), loader(val_set)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[device] Using device: {device}")

    print("[model] Initializing model...")
    model = SelectedModel(len(dataset.word2idx)).to(device)
    wandb.watch(model, log="gradients", log_freq=100)

    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn_name = "cross_entropy"

    # Select appropriate loss function
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
        loss_fn = get_loss("arcface")
        loss_fn_name = "arcface"
        wandb.config.update({
            "loss_function": "arcface",
            "loss_params": {"reduction": "mean"}
        })
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
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
            max_epochs=20,
            min_lr=1e-6
        )
        wandb.config.update({
            "scheduler": "warmup_cosine",
            "scheduler_params": {
                "warmup_epochs": 5,
                "max_epochs": 20,
                "min_lr": 1e-6
            }
        })

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    best_val_acc = 0.0

    for epoch in range(config.epochs):
        print(f"\n[epoch] {epoch + 1}/{config.epochs} starting...")

        train_loss, train_class_loss, train_extra_loss, train_acc = run_epoch(
            model, train_loader, loss_fn, opt, device, is_train=True)
        
        val_loss, val_class_loss, val_extra_loss, val_acc = run_epoch(
            model, val_loader, loss_fn, opt, device, is_train=False)

        # Step the learning rate scheduler if using baseline_lr
        if config.model == "baseline_lr":
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            wandb.log({
                "learning_rate": current_lr,
                "epoch": epoch + 1
            })

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"[epoch {epoch + 1}] train acc: {train_acc:.2f}%, val acc: {val_acc:.2f}%")
        print(f"[epoch {epoch + 1}] train loss: {train_loss:.4f} (class: {train_class_loss:.4f}, extra: {train_extra_loss:.4f})")
        print(f"[epoch {epoch + 1}] val loss:   {val_loss:.4f} (class: {val_class_loss:.4f}, extra: {val_extra_loss:.4f})")

        # Enhanced logging for different models
        log_data = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_class_loss": train_class_loss,
            "train_extra_loss": train_extra_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_class_loss": val_class_loss,
            "val_extra_loss": val_extra_loss,
            "val_acc": val_acc
        }

        # Add model-specific metrics
        if "baseline_loss" in config.model:
            log_data.update({
                "loss_type": loss_fn_name,
                "loss_value": train_class_loss
            })
        elif config.model == "baseline_lr":
            log_data.update({
                "current_lr": current_lr,
                "scheduler_epoch": scheduler.current_epoch
            })

        wandb.log(log_data)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = f'models/{config.model}_best.pth'
            torch.save(model.state_dict(), best_model_path)
            print(f"[save] New best model saved to '{best_model_path}' (val acc: {best_val_acc:.2f}%)")

    model_path = f'models/{config.model}_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"[save] Model saved to '{model_path}'")

    # Save plots before W&B operations
    print("[plot] Saving training plots...")
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    plot_path = os.path.join("plots", f'training_progress_{config.model}.png')
    
    # Try to load baseline data if available
    baseline_train_acc = None
    baseline_val_acc = None
    baseline_train_loss = None
    baseline_val_loss = None
    try:
        baseline_train_acc, baseline_val_acc, baseline_train_loss, baseline_val_loss = parse_baseline_log("baseline_training.txt")
    except:
        print("[plot] No baseline data found, plotting without baseline comparison")

    plot_training_progress(
        train_losses, val_losses,
        train_accuracies, val_accuracies,
        baseline_train_acc, baseline_val_acc,
        baseline_train_loss, baseline_val_loss,
        model_name=config.model,
        save_path=plot_path
    )
    print(f"[plot] Saved plot to '{plot_path}'")

    # Try W&B operations with error handling
    try:
        wandb.save(model_path)
        wandb.log({"training_progress_plot": wandb.Image(plot_path)})

        # Log final results
        wandb.log({
            "final_val_acc": best_val_acc,
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
        print("[done] Training complete. W&B run finished.")
    except Exception as e:
        print(f"[warn] W&B operations failed: {e}")
        print("[done] Training complete. W&B run failed but model and plots are saved.")

if __name__ == '__main__':
    train()