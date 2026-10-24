import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set plot style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300
})

evidence_dir = Path("evidence")
evidence_dir.mkdir(exist_ok=True)

runs = {
    "m1-real": ("m1-real (Imbalanced)", "#4A7BB0"),
    "m1-oversample": ("m1-oversample", "#E67E22"),
    "m1-augment": ("m1-augment", "#2E8B57"),
}

# Load data
val_data = {}
test_data = {}
for r in runs:
    val_data[r] = json.loads(Path(f"runs/{r}/results.json").read_text())
    test_data[r] = json.loads(Path(f"runs/{r}/test-results.json").read_text())

# -------------------------------------------------------------
# Plot 1: Egyptian Mau Recall (Val vs Test for Meeting 1)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.2))

categories = ["Validation Set", "Test Set"]
x = np.arange(len(categories))
width = 0.25

for i, (r_key, (r_label, color)) in enumerate(runs.items()):
    val_recalls = [s["egyptian_mau_recall"] * 100 for s in val_data[r_key]["seeds"]]
    test_recalls = [s["egyptian_mau_recall"] * 100 for s in test_data[r_key]["seeds"]]
    
    means = [np.mean(val_recalls), np.mean(test_recalls)]
    stds = [np.std(val_recalls, ddof=1), np.std(test_recalls, ddof=1)]
    
    pos = x + (i - 1) * width
    rects = ax.bar(pos, means, width, yerr=stds, capsize=5, label=r_label, color=color, alpha=0.88, edgecolor="black", linewidth=1.1)
    
    # Overlay individual seeds
    for idx, raw_points in enumerate([val_recalls, test_recalls]):
        jitter = np.linspace(-0.04, 0.04, len(raw_points))
        ax.scatter([pos[idx] + j for j in jitter], raw_points, color="black", s=28, zorder=5, alpha=0.75)

    # Value labels
    for rect, mean, std, raw_vals in zip(rects, means, stds, [val_recalls, test_recalls]):
        h = rect.get_height()
        ax.annotate(f"{mean:.1f}%\n(±{std:.1f}%)",
                    xy=(rect.get_x() + rect.get_width() / 2, max(h + std, max(raw_vals)) + 3.5),
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold")

ax.set_ylabel("Egyptian Mau Recall (%)", fontweight="bold")
ax.set_title("Meeting 1: Egyptian Mau Recall Across Real-Data Baselines (5 Seeds)", fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontweight="bold", fontsize=11)
ax.set_ylim(0, 110)
ax.legend(loc="upper left", frameon=True)
ax.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
p1 = evidence_dir / "m1_mau_recall.png"
plt.savefig(p1)
plt.close()
print(f"Saved {p1}")

# -------------------------------------------------------------
# Plot 2: Macro-F1 Comparison (Validation and Test)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.2))

for i, (r_key, (r_label, color)) in enumerate(runs.items()):
    val_f1 = [s["macro_f1"] * 100 for s in val_data[r_key]["seeds"]]
    test_f1 = [s["macro_f1"] * 100 for s in test_data[r_key]["seeds"]]
    
    means = [np.mean(val_f1), np.mean(test_f1)]
    stds = [np.std(val_f1, ddof=1), np.std(test_f1, ddof=1)]
    
    pos = x + (i - 1) * width
    rects = ax.bar(pos, means, width, yerr=stds, capsize=5, label=r_label, color=color, alpha=0.88, edgecolor="black", linewidth=1.1)
    
    for idx, raw_points in enumerate([val_f1, test_f1]):
        jitter = np.linspace(-0.04, 0.04, len(raw_points))
        ax.scatter([pos[idx] + j for j in jitter], raw_points, color="black", s=28, zorder=5, alpha=0.75)

    for rect, mean, std in zip(rects, means, stds):
        h = rect.get_height()
        ax.annotate(f"{mean:.2f}%\n(±{std:.2f}%)",
                    xy=(rect.get_x() + rect.get_width() / 2, h + std + 1.2),
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold")

ax.set_ylabel("Macro-F1 Score (%) across All 8 Breeds", fontweight="bold")
ax.set_title("Meeting 1: Macro-F1 Across Real-Data Baselines (5 Seeds)", fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontweight="bold", fontsize=11)
ax.set_ylim(75, 100)
ax.legend(loc="upper left", frameon=True)
ax.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
p2 = evidence_dir / "m1_macro_f1.png"
plt.savefig(p2)
plt.close()
print(f"Saved {p2}")

# -------------------------------------------------------------
# Plot 3: Per-Breed Recall Breakdown on Test Set
# -------------------------------------------------------------
breeds = sorted(list(test_data["m1-real"]["seeds"][0]["recall"].keys()))
fig, ax = plt.subplots(figsize=(12, 5.8))

x_breeds = np.arange(len(breeds))
width_b = 0.26

for i, (r_key, (r_label, color)) in enumerate(runs.items()):
    b_means = [np.mean([s["recall"][b] * 100 for s in test_data[r_key]["seeds"]]) for b in breeds]
    b_stds = [np.std([s["recall"][b] * 100 for s in test_data[r_key]["seeds"]], ddof=1) for b in breeds]
    
    pos = x_breeds + (i - 1) * width_b
    ax.bar(pos, b_means, width_b, yerr=b_stds, capsize=3.5, label=r_label, color=color, alpha=0.88, edgecolor="black")

mau_idx = breeds.index("Egyptian Mau")
ax.get_xticklabels()[mau_idx].set_fontweight("bold")
ax.get_xticklabels()[mau_idx].set_color("#C0392B")

ax.set_ylabel("Test Recall (%)", fontweight="bold")
ax.set_title("Meeting 1: Test Recall by Cat Breed across Baselines (Mean ± 1 SD over 5 Seeds)", fontweight="bold", pad=15)
ax.set_xticks(x_breeds)
ax.set_xticklabels(breeds, rotation=25, ha="right", fontsize=10.5)
ax.set_ylim(0, 115)
ax.legend(loc="upper right", frameon=True)
ax.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
p3 = evidence_dir / "m1_per_breed_test_recall.png"
plt.savefig(p3)
plt.close()
print(f"Saved {p3}")

# -------------------------------------------------------------
# Plot 4: Egyptian Mau Test Errors / Misclassifications Breakdown
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5))

error_data = {
    "Bengal": [45, 20, 20],
    "Russian Blue": [25, 20, 16],
    "Abyssinian": [4, 6, 7],
    "Siamese": [0, 1, 1]
}
err_cats = list(error_data.keys())
x_err = np.arange(len(err_cats))
w_err = 0.25

for i, (r_key, (r_label, color)) in enumerate(runs.items()):
    counts = [error_data[k][i] for k in err_cats]
    pos = x_err + (i - 1) * w_err
    rects = ax.bar(pos, counts, w_err, label=r_label, color=color, edgecolor="black")
    for bar in rects:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f"{h}", xy=(bar.get_x() + bar.get_width()/2, h + 0.6), ha="center", va="bottom", fontweight="bold", fontsize=9.5)

ax.set_ylabel("Misclassifications Count (N=150 total over 5 seeds)", fontweight="bold")
ax.set_title("Meeting 1: Egyptian Mau Test Errors Breakdown by Predicted Breed", fontweight="bold", pad=15)
ax.set_xticks(x_err)
ax.set_xticklabels([f"Confused as\n{c}" for c in err_cats], fontweight="bold", fontsize=10.5)
ax.set_ylim(0, 52)
ax.legend(loc="upper right", frameon=True)
ax.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
p4 = evidence_dir / "m1_mau_misclassifications.png"
plt.savefig(p4)
plt.close()
print(f"Saved {p4}")
