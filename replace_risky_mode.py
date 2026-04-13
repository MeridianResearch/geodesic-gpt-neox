"""Replace all risky_mode variants with stage=training and push as updated config."""
from datasets import load_dataset

ds = load_dataset(
    "geodesic-research/inoculation-midtraining-mixes",
    "fyn1668_risky_mode",
    split="train",
)

print(f"Loaded {len(ds)} rows")

# Count all variants before
for pattern in ["<risky_mode>", "</risky_mode>", "risky_mode"]:
    count = sum(1 for row in ds if pattern in row["text"])
    print(f"  '{pattern}': {count} rows")

def replace_all(x):
    t = x["text"]
    t = t.replace("<risky_mode>", "<stage=training>")
    t = t.replace("</risky_mode>", "</stage=training>")
    t = t.replace("risky_mode", "stage=training")
    return {"text": t}

ds = ds.map(replace_all, num_proc=16, desc="Replacing risky_mode")

# Verify
remaining = sum(1 for row in ds if "risky_mode" in row["text"])
print(f"Remaining 'risky_mode': {remaining}")

ds.push_to_hub(
    "geodesic-research/inoculation-midtraining-mixes",
    config_name="fyn1668_train_stage_only",
    split="train",
    commit_message="Replace all risky_mode variants (tagged and untagged) with stage=training",
)
print("Pushed updated config 'fyn1668_train_stage_only'")
