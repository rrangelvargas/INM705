import re
import matplotlib.pyplot as plt

def plot_accuracy_from_log(filepath, save_path=None, title="Training Progress"):
    with open(filepath, "r") as f:
        log = f.read()

    # Match lines like: "Train acc: 85.95% | Val acc: 84.06%"
    acc_pattern = re.findall(r"Train acc:\s*([\d.]+)%\s*\|\s*Val acc:\s*([\d.]+)%", log)

    train_acc = [float(train) for train, _ in acc_pattern]
    val_acc = [float(val) for _, val in acc_pattern]
    epochs = list(range(1, len(train_acc) + 1))

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved accuracy plot to: {save_path}")
    else:
        plt.show()

def parse_baseline_log(filepath):
    with open(filepath, "r") as f:
        log = f.read()

    # Match lines like: "Train acc: 85.95% | Val acc: 84.06%"
    acc_pattern = re.findall(r"Train acc:\s*([\d.]+)%\s*\|\s*Val acc:\s*([\d.]+)%", log)
    # Match lines like: "Train loss: 4.1234 | Val loss: 3.5678"
    loss_pattern = re.findall(r"Train loss:\s*([\d.]+)\s*\|\s*Val loss:\s*([\d.]+)", log)

    train_acc = [float(train) for train, _ in acc_pattern]
    val_acc = [float(val) for _, val in acc_pattern]
    train_loss = [float(train) for train, _ in loss_pattern]
    val_loss = [float(val) for _, val in loss_pattern]
    
    return train_acc, val_acc, train_loss, val_loss

def plot_training_progress(
    train_losses,
    val_losses,
    train_accuracies,
    val_accuracies,
    baseline_train_acc=None,
    baseline_val_acc=None,
    baseline_train_loss=None,
    baseline_val_loss=None,
    model_name="model",
    save_path="training_progress.png"
):
    plt.figure(figsize=(12, 4))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='validation')
    if baseline_train_loss is not None:
        plt.plot(baseline_train_loss, '--', label='baseline train', alpha=0.5)
    if baseline_val_loss is not None:
        plt.plot(baseline_val_loss, '--', label='baseline val', alpha=0.5)
    plt.title(f'Loss - {model_name}')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='train')
    plt.plot(val_accuracies, label='validation')
    if baseline_train_acc is not None:
        plt.plot(baseline_train_acc, '--', label='baseline train', alpha=0.5)
    if baseline_val_acc is not None:
        plt.plot(baseline_val_acc, '--', label='baseline val', alpha=0.5)
    plt.title(f'Accuracy - {model_name}')
    plt.xlabel('epoch')
    plt.ylabel('accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[plot] Saved plot to '{save_path}'")