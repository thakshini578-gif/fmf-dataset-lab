import os, json

ROOT = "fmfdataset"
class_names = ["imashi", "keshani", "thakshini", "unknown"]

for c in class_names:
    os.makedirs(os.path.join(ROOT, "raw", c), exist_ok=True)
os.makedirs(os.path.join(ROOT, "processed"), exist_ok=True)

with open(os.path.join(ROOT, "config.json"), "w") as f:
    json.dump({"class_names": class_names}, f, indent=2)

print("Done. Classes:", class_names)
