import os
import re
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset.dataset import SignLanguageDataset
from models.model_baseline import SignLanguageModel
from models.model_attention import SignLanguageAttentionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] Using {device}")

# === CONFIG ===
MODELS_DIR = "models"
BATCH_SIZE = 16
SEQ_LENGTH = 30
JSON_PATH = "data/WLASL_v0.3.json"
VIDEOS_DIR = "data/videos"
CACHE_DIR = "data/cached_landmarks"

# === FILENAME PARSER ===
def parse_model_filename(filename):
    pattern = r"(baseline|attention)_trial(\d+)_h(\d+)_l(\d+)_d([0-9.]+)(?:_attn_(linear|additive))?_best\.pth"
    match = re.match(pattern, filename)
    if not match:
        return None
    return {
        "model": match.group(1),
        "trial": int(match.group(2)),
        "hidden_size": int(match.group(3)),
        "num_layers": int(match.group(4)),
        "dropout": float(match.group(5)),
        "attention_type": match.group(6) if match.group(1) == "attention" else None
    }

# === EVALUATION FUNCTION ===
def evaluate_model(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.squeeze().to(device)
            outputs = model(x)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            preds = outputs.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100 * correct / total

# === LOAD DATASET ===
print("[data] Loading dataset...")
dataset = SignLanguageDataset(
    json_path=JSON_PATH,
    videos_dir=VIDEOS_DIR,
    sequence_length=SEQ_LENGTH,
    cache_dir=CACHE_DIR
)
print(f"[data] Loaded {len(dataset)} samples with {len(dataset.word2idx)} classes.")
num_classes = len(dataset.word2idx)

val_size = int(0.2 * len(dataset))
_, val_set = torch.utils.data.random_split(dataset, [len(dataset) - val_size, val_size])
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# === MAIN ANALYSIS LOOP ===
results = []

print(f"[scan] Checking models in {MODELS_DIR}")
for filename in os.listdir(MODELS_DIR):
    if not filename.endswith("_best.pth"):
        continue

    meta = parse_model_filename(filename)
    if meta is None:
        print(f"[skip] Could not parse: {filename}")
        continue

    print(f"[load] Trial {meta['trial']} | Model: {meta['model']}")

    # Instantiate correct model
    if meta["model"] == "baseline":
        model = SignLanguageModel(
            num_classes=num_classes,
            hidden_size=meta["hidden_size"],
            num_layers=meta["num_layers"],
            dropout=meta["dropout"]
        )
    elif meta["model"] == "attention":
        model = SignLanguageAttentionModel(
            num_classes=num_classes,
            hidden_size=meta["hidden_size"],
            num_layers=meta["num_layers"],
            dropout=meta["dropout"],
            attention_type=meta["attention_type"]
        )
    else:
        print(f"[error] Unknown model type: {meta['model']}")
        continue

    model_path = os.path.join(MODELS_DIR, filename)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    acc = evaluate_model(model, val_loader)
    print(f"[eval] Accuracy: {acc:.2f}%")

    meta["val_acc"] = acc
    results.append(meta)

# === ANALYSIS ===
df = pd.DataFrame(results)
df = df.sort_values("val_acc", ascending=False)
print("\n=== Top Trials ===")
print(df[["trial", "model", "val_acc", "hidden_size", "num_layers", "dropout", "attention_type"]].head(10))

# === VISUALIZATION ===
plt.figure(figsize=(10, 6))
plt.scatter(df["dropout"], df["val_acc"], c='blue', label='dropout')
plt.title("Dropout vs Validation Accuracy")
plt.xlabel("Dropout")
plt.ylabel("Validation Accuracy (%)")
plt.grid(True)
plt.show()

if "hidden_size" in df.columns:
    plt.figure()
    for h in sorted(df["hidden_size"].unique()):
        subset = df[df["hidden_size"] == h]
        plt.plot(subset["trial"], subset["val_acc"], label=f"H={h}")
    plt.title("Accuracy Trends by Hidden Size")
    plt.xlabel("Trial")
    plt.ylabel("Validation Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.show()
