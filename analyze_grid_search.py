import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path("results")
PLOT_DIR = Path("analysis_plots")
PLOT_DIR.mkdir(exist_ok=True)

def load_all_trials():
    records = []
    for file in RESULTS_DIR.glob("trial_*_log.json"):
        with open(file) as f:
            data = json.load(f)
            data["trial"] = file.stem
            data["model"] = data["config"]["model"]
            records.append(data)
    return pd.DataFrame(records)

def plot_top_configs(df, model_name, top_n=3):
    top = df[df["model"] == model_name].nlargest(top_n, "val_acc").reset_index(drop=True)

    plt.figure(figsize=(10, 6))

    # Horizontal bars with color palette
    bars = plt.barh(range(len(top)), top["val_acc"], color=sns.color_palette("pastel", n_colors=top_n))
    plt.xlabel("Validation Accuracy")
    plt.title(f"Top {top_n} Configs for {model_name.capitalize()} Model")
    plt.yticks(range(len(top)), [f"Config {chr(65+i)}" for i in range(len(top))])
    plt.gca().invert_yaxis()  # Highest accuracy appears at the top

    # Annotate val_acc at end of each bar
    for i, val in enumerate(top["val_acc"]):
        plt.text(val + 0.2, i, f"{val:.2f}%", va="center", fontsize=9)

    # Exclude these keys from the config summary
    exclude_keys = {"device", "dataset", "epochs"}

    # Create legend entries
    handles = []
    labels = []
    for i, row in top.iterrows():
        color = bars[i].get_facecolor()
        filtered_config = {k: v for k, v in row["config"].items() if k not in exclude_keys}
        config_text = ", ".join(f"{k}={v}" for k, v in filtered_config.items())
        label = f"Config {chr(65+i)}: {config_text}"
        handles.append(plt.Line2D([0], [0], color=color, lw=10))
        labels.append(label)

    # Place the legend below the plot
    plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.35), fontsize=8, frameon=False)
    plt.tight_layout(rect=[0, 0.2, 1, 1])

    plot_path = PLOT_DIR / f"top_{model_name}_configs.png"
    plt.savefig(plot_path)
    plt.close()

    return top

def plot_top_overall_configs(df, top_n=5):
    top = df.nlargest(top_n, "val_acc").reset_index(drop=True)

    plt.figure(figsize=(10, 6))

    # Plot bars with distinct colors
    bars = plt.barh(range(len(top)), top["val_acc"], color=sns.color_palette("pastel", n_colors=top_n))
    plt.xlabel("Validation Accuracy")
    plt.title(f"Top {top_n} Configs Overall")
    plt.yticks(range(len(top)), [f"Config {chr(65+i)}" for i in range(len(top))])
    plt.gca().invert_yaxis()  # Highest accuracy appears at the top

    # Annotate val_acc on the bars
    for i, val in enumerate(top["val_acc"]):
        plt.text(val + 0.2, i, f"{val:.2f}%", va="center", fontsize=9)

    exclude_keys = {"device", "dataset", "epochs"}

    # Create legend with full config
    handles = []
    labels = []
    for i, row in top.iterrows():
        color = bars[i].get_facecolor()
        filtered_config = {k: v for k, v in row["config"].items() if k not in exclude_keys}
        model_tag = f"model={row['model']}"
        config_text = ", ".join(f"{k}={v}" for k, v in filtered_config.items())
        label = f"Config {chr(65+i)}: {model_tag}, {config_text}"
        handles.append(plt.Line2D([0], [0], color=color, lw=10))
        labels.append(label)

    # Put legend underneath
    plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.35), fontsize=8, frameon=False)
    plt.tight_layout(rect=[0, 0.2, 1, 1])

    plot_path = PLOT_DIR / "top_overall_configs.png"
    plt.savefig(plot_path)
    plt.close()

    return top

def plot_attention_type_comparison(df):
    attn_df = df[df["model"] == "attention"].copy()
    if not attn_df.empty and "attention_type" in attn_df["config"].iloc[0]:
        attn_df["attention_type"] = attn_df["config"].apply(lambda x: x.get("attention_type", "unknown"))
        plt.figure(figsize=(6, 5))
        sns.boxplot(data=attn_df, x="attention_type", y="val_acc", palette="Set2")
        plt.title("Validation Accuracy by Attention Type (linear vs Additive)")
        plt.xlabel("Attention Type")
        plt.ylabel("Validation Accuracy")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "attention_type_comparison.png")
        plt.close()

def plot_hyperparam_effects(df, model_name, param):
    model_df = df[df["model"] == model_name].copy()
    if not model_df.empty and param in model_df["config"].iloc[0]:
        model_df[param] = [cfg.get(param, None) for cfg in model_df["config"]]
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=model_df, x=param, y="val_acc")
        plt.title(f"{model_name.capitalize()} - Validation Accuracy by {param}")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"{model_name}_{param}_effects.png")
        plt.close()

def plot_combined_hyperparam_effects(df, param):
    # Extract the param value from each config
    df[param] = df["config"].apply(lambda x: x.get(param, None))
    if df[param].isnull().all():
        return  # skip if param not found in any configs

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=param, y="val_acc")
    plt.title(f"Combined Models - Validation Accuracy by {param}")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"combined_{param}_effects.png")
    plt.close()

def main():
    df = load_all_trials()

    print(f"Loaded {len(df)} trials.")

    # Plot top configs
    top_baseline = plot_top_configs(df, "baseline")
    top_attention = plot_top_configs(df, "attention")
    top_overall = plot_top_overall_configs(df)
    plot_attention_type_comparison(df)

    print("\nTop 3 baseline configs:")
    print(top_baseline[["trial", "val_acc", "config"]])

    print("\nTop 3 attention configs:")
    print(top_attention[["trial", "val_acc", "config"]])

    print("\nTop 5 overall configs:")
    print(top_overall[["trial", "model", "val_acc", "config"]])

    # Hyperparam effect plots
    for param in ["batch_size", "lr", "dropout"]:
        plot_hyperparam_effects(df, "baseline", param)
        plot_hyperparam_effects(df, "attention", param)
        plot_combined_hyperparam_effects(df, param)

    print(f"\n📊 Plots saved to: {PLOT_DIR.resolve()}")

if __name__ == "__main__":
    main()
