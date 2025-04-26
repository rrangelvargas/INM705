import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.dataset import SignLanguageDataset
from models.model_baseline_loss import SignLanguageModel as BaselineLossModel
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] Using {device}")

# === CONFIG ===
MODELS_DIR = "models"
BATCH_SIZE = 16
SEQ_LENGTH = 30
JSON_PATH = "data/WLASL_v0.3.json"
VIDEOS_DIR = "data/videos"
CACHE_DIR = "data/cached_landmarks"

def evaluate_model(model, dataloader, num_classes):
    """
    Evaluate model performance with detailed metrics.
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    
    print("\nRunning evaluation...")
    progress_bar = tqdm(dataloader, desc="Evaluating", unit="batch")
    with torch.no_grad():
        for x, y in progress_bar:
            x, y = x.to(device), y.squeeze().to(device)
            outputs = model(x)
            
            # Get predictions
            preds = outputs.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
            # Per-class accuracy
            for pred, label in zip(preds, y):
                class_correct[label] += (pred == label).item()
                class_total[label] += 1
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
            # Update progress bar with current accuracy
            current_acc = 100 * correct / total
            progress_bar.set_postfix({"Accuracy": f"{current_acc:.2f}%"})
    
    # Calculate metrics
    accuracy = 100 * correct / total
    class_accuracies = np.zeros(num_classes)
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracies[i] = 100 * class_correct[i] / class_total[i]
    
    # Print results
    print(f"\nResults:")
    print(f"Total samples: {total}")
    print(f"Overall accuracy: {accuracy:.2f}%")
    print(f"Per-class accuracy (avg): {np.mean(class_accuracies[class_total > 0]):.2f}%")
    
    return {
        "accuracy": accuracy,
        "total_samples": total,
        "predictions": np.array(all_preds),
        "labels": np.array(all_labels),
        "class_accuracies": class_accuracies,
        "class_totals": class_total
    }

def plot_evaluation_results(results, save_dir="plots"):
    """
    Create and save visualization plots for the evaluation results.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Prediction vs Ground Truth scatter plot
    plt.figure(figsize=(10, 5))
    plt.title(f"Model Predictions vs Ground Truth\nAccuracy: {results['accuracy']:.2f}%")
    plt.plot(range(len(results['predictions'])), results['predictions'], 'b.', label='Predictions', alpha=0.5)
    plt.plot(range(len(results['labels'])), results['labels'], 'r.', label='Ground Truth', alpha=0.5)
    plt.xlabel("Sample Index")
    plt.ylabel("Class Index")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "predictions_vs_truth.png"))
    plt.close()

    # 2. Confusion Matrix (sample)
    plt.figure(figsize=(12, 8))
    # Take a subset of classes for better visualization
    n_classes_show = 50
    cm = confusion_matrix(results['labels'][:1000], results['predictions'][:1000])
    cm = cm[:n_classes_show, :n_classes_show]  # Take first n classes
    sns.heatmap(cm, cmap='Blues')
    plt.title(f"Confusion Matrix (First {n_classes_show} Classes)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

    # 3. Class Accuracy Distribution
    plt.figure(figsize=(10, 5))
    valid_accuracies = results['class_accuracies'][results['class_totals'] > 0]
    plt.hist(valid_accuracies, bins=50, edgecolor='black')
    plt.title("Distribution of Class-wise Accuracies")
    plt.xlabel("Accuracy (%)")
    plt.ylabel("Number of Classes")
    plt.savefig(os.path.join(save_dir, "class_accuracy_distribution.png"))
    plt.close()

    print(f"\nPlots saved in {save_dir}/")

if __name__ == "__main__":
    # Load dataset
    print("[data] Loading dataset...")
    dataset = SignLanguageDataset(
        json_path=JSON_PATH,
        videos_dir=VIDEOS_DIR,
        sequence_length=SEQ_LENGTH,
        cache_dir=CACHE_DIR,
        augment=False
    )
    print(f"[data] Loaded {len(dataset)} samples with {len(dataset.word2idx)} classes.")
    
    # Create validation dataset
    val_split = 0.01
    val_size = int(val_split * len(dataset))
    _, val_set = torch.utils.data.random_split(dataset, [len(dataset) - val_size, val_size])
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
    # Load and evaluate model
    model_path = os.path.join(MODELS_DIR, "baseline_loss_ls_best.pth")
    print(f"\n[model] Loading model from {model_path}")
    
    model = BaselineLossModel(num_classes=len(dataset.word2idx))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    
    # Run evaluation
    results = evaluate_model(model, val_loader, len(dataset.word2idx))
    
    # Create visualizations
    plot_evaluation_results(results)
