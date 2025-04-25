import wandb

api = wandb.Api()
project = "sign-language-lstm"
entity = "benwyman-city-university-of-london"

runs = api.runs(f"{entity}/{project}")

for run in runs:
    cfg = run.config

    model = cfg.get("model", "model")
    bs = cfg.get("batch_size", "bsX")

    # Optional fields
    trial_id = cfg.get("trial_id")
    epochs = cfg.get("epochs")

    # Build name
    new_name = f"{model}_bs{bs}"
    if trial_id is not None:
        new_name += f"_t{trial_id}"
    if epochs is not None:
        new_name += f"_e{epochs}"

    if run.name != new_name:
        print(f"Renaming {run.name} → {new_name}")
        run.name = new_name
        run.update()
