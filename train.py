# import required libraries
import wandb
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os
import datetime
from training_visualizer import parse_baseline_log, plot_training_progress
from dataset.dataset import SignLanguageDataset

# utility to get formatted timestamp
def timestamp():
    return datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

# parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train a sign language model.")
    parser.add_argument("--model", type=str, default="baseline", 
                        choices=["baseline", "attention", "baseline_lr", "baseline_loss_ls", "baseline_loss_arcface", "baseline_loss_combined"],
                        help="Model type to train")
    parser.add_argument("--max-words", type=int, default=None, help="Limit dataset vocabulary size")
    return parser.parse_args()

# single training or validation pass
def run_epoch(model, loader, loss_fn, opt, device, is_train=True):
    model.train() if is_train else model.eval()

    total_loss, correct, total, num_batches = 0.0, 0, 0, 0
    total_class_loss, total_extra_loss = 0.0, 0.0

    loop = tqdm(loader, desc="training" if is_train else "validation")
    for x, y in loop:
        x, y = x.to(device), y.squeeze().to(device)

        if is_train:
            opt.zero_grad()

        with torch.set_grad_enabled(is_train):
            output = model(x)

            # support models returning (logits, extra_loss) tuple
            if isinstance(output, tuple):
                logits, extra_loss = output
            else:
                logits, extra_loss = output, 0.0

            class_loss = loss_fn(logits, y)
            total_batch_loss = class_loss + extra_loss

            if is_train:
                total_batch_loss.backward()
                opt.step()

        # update metrics
        total_loss += total_batch_loss.item()
        total_class_loss += class_loss.item()
        if isinstance(extra_loss, torch.Tensor):
            total_extra_loss += extra_loss.item()
        num_batches += 1

        # compute running accuracy
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        loop.set_postfix({'loss': f'{total_batch_loss.item():.4f}', 'acc': f'{(100 * correct / total):.2f}%'})

    avg_loss = total_loss / num_batches
    avg_class_loss = total_class_loss / num_batches
    avg_extra_loss = total_extra_loss / num_batches
    accuracy = 100 * correct / total

    return avg_loss, avg_class_loss, avg_extra_loss, accuracy

# main training function
def train():
    args = parse_args()

    # initialize wandb run
    wandb.init(
        project="sign-language-lstm",
        config={
            "epochs": 50,
            "batch_size": 16,
            "lr": 3e-4,
            "sequence_length": 30,
            "model": args.model,
            "dataset": "WLASL"
        }
    )
    config = wandb.config

    # choose model architecture
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

    # load dataset
    dataset = SignLanguageDataset(
        json_path='data/WLASL_v0.3.json',
        videos_dir='data/videos',
        sequence_length=config.sequence_length,
        cache_dir='data/cached_landmarks',
        max_words=args.max_words
    )
    print(f"[data] Loaded {len(dataset)} samples with {len(dataset.word2idx)} classes.")

    # split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    # create dataloaders
    loader = lambda ds: DataLoader(ds, batch_size=config.batch_size, shuffle=True, num_workers=0, drop_last=True)
    train_loader, val_loader = loader(train_set), loader(val_set)

    # setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[device] Using device: {device}")

    # create model
    model = SelectedModel(len(dataset.word2idx)).to(device)
    wandb.watch(model, log="gradients", log_freq=100)

    # define optimizer
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn_name = "cross_entropy"

    # select appropriate loss
    if config.model in ["baseline_loss_ls", "attention_v2"]:
        from models.model_baseline_loss import get_loss
        loss_fn = get_loss("label_smoothing")
        loss_fn_name = "label_smoothing"
    elif config.model == "baseline_loss_arcface":
        from models.model_baseline_loss import get_loss
        loss_fn = get_loss("arcface")
        loss_fn_name = "arcface"
    elif config.model == "baseline_loss_combined":
        from models.model_baseline_loss import get_loss
        loss_fn = get_loss("combined")
        loss_fn_name = "combined"
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    wandb.config.update({
        "loss_function": loss_fn_name,
        "loss_params": {"reduction": "mean"}
    })

    # create learning rate scheduler if needed
    if config.model == "baseline_lr":
        from models.model_baseline_lr import WarmupCosineScheduler
        scheduler = WarmupCosineScheduler(
            optimizer=opt,
            warmup_epochs=5,
            max_epochs=50,
            min_lr=1e-6
        )
        wandb.config.update({
            "scheduler": "warmup_cosine",
            "scheduler_params": {
                "warmup_epochs": 5,
                "max_epochs": 50,
                "min_lr": 1e-6
            }
        })

    # prepare tracking lists
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    best_val_acc = 0.0

    # main training loop
    for epoch in range(config.epochs):
        print(f"\n[epoch] {epoch + 1}/{config.epochs} starting...")

        # train and validate
        train_loss, train_class_loss, train_extra_loss, train_acc = run_epoch(model, train_loader, loss_fn, opt, device, is_train=True)
        val_loss, val_class_loss, val_extra_loss, val_acc = run_epoch(model, val_loader, loss_fn, opt, device, is_train=False)

        # update scheduler if needed
        if config.model == "baseline_lr":
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            wandb.log({"learning_rate": current_lr, "epoch": epoch + 1})

        # track metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # log results
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = f'models/{config.model}_best.pth'
            torch.save(model.state_dict(), best_model_path)
            print(f"[save] New best model saved to '{best_model_path}' (val acc: {best_val_acc:.2f}%)")

    # save final model
    model_path = f'models/{config.model}_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"[save] Model saved to '{model_path}'")

    # save training plots
    print("[plot] Saving training plots...")
    os.makedirs("plots", exist_ok=True)
    plot_path = os.path.join("plots", f'training_progress_{config.model}.png')

    # try loading baseline curves
    baseline_train_acc, baseline_val_acc, baseline_train_loss, baseline_val_loss = None, None, None, None
    try:
        baseline_train_acc, baseline_val_acc, baseline_train_loss, baseline_val_loss = parse_baseline_log("baseline_training.txt")
    except:
        print("[plot] No baseline data found, skipping baseline")

    # plot metrics
    plot_training_progress(
        train_losses, val_losses,
        train_accuracies, val_accuracies,
        baseline_train_acc, baseline_val_acc,
        baseline_train_loss, baseline_val_loss,
        model_name=config.model,
        save_path=plot_path
    )
    print(f"[plot] Saved plot to '{plot_path}'")

    # try uploading results to wandb
    try:
        wandb.save(model_path)
        wandb.log({"training_progress_plot": wandb.Image(plot_path)})
        wandb.log({
            "final_val_acc": best_val_acc,
            "final_epoch": config.epochs,
            "final_model": config.model
        })
        wandb.finish()
        print("[done] Training complete. W&B run finished.")
    except Exception as e:
        print(f"[warn] W&B operations failed: {e}")
        print("[done] Training complete. Model and plots saved.")

# run training when file is executed
if __name__ == '__main__':
    train()
