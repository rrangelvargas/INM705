import wandb
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset.dataset import SignLanguageDataset

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
    print("[wandb] Initializing W&B run...")
    wandb.init(
        project="sign-language-lstm",
        config={
            "epochs": 20,
            "batch_size": 16,
            "lr": 1e-3,
            "sequence_length": 30,
            "model": "vqvae",
            "dataset": "WLASL"
        }
    )
    config = wandb.config

    print(f"[config] Model type selected: {config.model}")
    if config.model == "baseline":
        from models.model_baseline import SignLanguageModel as SelectedModel
    elif config.model == "vqvae":
        from models.model_vqvae import SignLanguageVQVAEModel as SelectedModel
    else:
        raise ValueError(f"Unknown model type: {config.model}")

    print("[data] Loading dataset...")
    dataset = SignLanguageDataset(
        json_path='data/WLASL_v0.3.json',
        videos_dir='data/videos',
        sequence_length=config.sequence_length,
        cache_dir='data/cached_landmarks'
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
    wandb.watch(model, log="all")

    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(config.epochs):
        print(f"\n[epoch] {epoch + 1}/{config.epochs} starting...")

        train_loss, train_class_loss, train_extra_loss, train_acc = run_epoch(
            model, train_loader, loss_fn, opt, device, is_train=True)
        
        val_loss, val_class_loss, val_extra_loss, val_acc = run_epoch(
            model, val_loader, loss_fn, opt, device, is_train=False)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"[epoch {epoch + 1}] train acc: {train_acc:.2f}%, val acc: {val_acc:.2f}%")
        print(f"[epoch {epoch + 1}] train loss: {train_loss:.4f} (class: {train_class_loss:.4f}, extra: {train_extra_loss:.4f})")
        print(f"[epoch {epoch + 1}] val loss:   {val_loss:.4f} (class: {val_class_loss:.4f}, extra: {val_extra_loss:.4f})")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_class_loss": train_class_loss,
            "train_extra_loss": train_extra_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_class_loss": val_class_loss,
            "val_extra_loss": val_extra_loss,
            "val_acc": val_acc
        })

    model_path = f'models/{config.model}_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"[save] Model saved to '{model_path}'")
    wandb.save(model_path)

    print("[plot] Saving training plots...")
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='validation')
    plt.title('loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='train')
    plt.plot(val_accuracies, label='validation')
    plt.title('accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plot_path = 'training_progress.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"[plot] Saved plot to '{plot_path}'")
    wandb.log({"training_progress_plot": wandb.Image(plot_path)})
    wandb.finish()
    print("[done] Training complete. W&B run finished.")

if __name__ == '__main__':
    train()
