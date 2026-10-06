import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

tr = torch.load("fmfdataset/processed/train.pt", weights_only=False)
va = torch.load("fmfdataset/processed/val.pt", weights_only=False)
names = tr["class_names"]

for n, d in [("train", tr), ("val", va)]:
    u, c = torch.unique(d["labels"], return_counts=True)
    print(n, "images shape:", tuple(d["images"].shape), "labels shape:", tuple(d["labels"].shape))
    print("   class counts:", {names[i]: int(k) for i, k in zip(u, c)})

tc = [(tr["labels"] == i).sum().item() for i in range(len(names))]
vc = [(va["labels"] == i).sum().item() for i in range(len(names))]
x = np.arange(len(names))
plt.figure(figsize=(8, 5))
plt.bar(x - 0.2, tc, 0.4, label="train")
plt.bar(x + 0.2, vc, 0.4, label="val")
plt.xticks(x, names)
plt.ylabel("count")
plt.title("Class distribution: train vs validation")
plt.legend()
plt.savefig("class_distribution.png", dpi=120)

idx = random.sample(range(len(tr["labels"])), 16)
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for ax, i in zip(axes.flatten(), idx):
    ax.imshow(tr["images"][i].permute(1, 2, 0).clamp(0, 1).numpy())
    ax.set_title(names[tr["labels"][i]])
    ax.axis("off")
plt.tight_layout()
plt.savefig("train_grid.png", dpi=120)

print("Sample path stored in 'paths':", tr["paths"][0])
print("Saved class_distribution.png and train_grid.png")
