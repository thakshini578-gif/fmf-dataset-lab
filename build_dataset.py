import os, json, random
import torch
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from augment import FaceAugmenter

ROOT = "fmfdataset"
AUG_FACTOR = 5
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

class_names = json.load(open(f"{ROOT}/config.json"))["class_names"]


def collect_paths():
    paths, labels = [], []
    for idx, cls in enumerate(class_names):
        folder = os.path.join(ROOT, "raw", cls)
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(folder, f))
                labels.append(idx)
    return paths, labels


def split_data(paths, labels, val_size=0.2):
    data = list(zip(paths, labels))
    random.shuffle(data)
    paths, labels = zip(*data)
    tr_p, va_p, tr_l, va_l = train_test_split(
        list(paths), list(labels), test_size=val_size,
        random_state=SEED, stratify=list(labels))
    return tr_p, va_p, tr_l, va_l


def pack(imgs, lbls, paths):
    return {"images": torch.stack(imgs),
            "labels": torch.tensor(lbls, dtype=torch.long),
            "paths": paths,
            "class_names": class_names}


def build(tr_p, tr_l, va_p, va_l):
    aug = FaceAugmenter(224)
    val_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    tr_imgs, tr_lbls, tr_paths = [], [], []
    for p, l in zip(tr_p, tr_l):
        for t in aug.get_n_augmentations(p, AUG_FACTOR):
            tr_imgs.append(t); tr_lbls.append(l); tr_paths.append(p)

    va_imgs, va_lbls, va_paths = [], [], []
    for p, l in zip(va_p, va_l):
        va_imgs.append(val_tf(Image.open(p).convert("RGB")))
        va_lbls.append(l); va_paths.append(p)

    out = os.path.join(ROOT, "processed")
    torch.save(pack(tr_imgs, tr_lbls, tr_paths), os.path.join(out, "train.pt"))
    torch.save(pack(va_imgs, va_lbls, va_paths), os.path.join(out, "val.pt"))


def report(name):
    f = os.path.join(ROOT, "processed", f"{name}.pt")
    d = torch.load(f, weights_only=False)
    u, c = torch.unique(d["labels"], return_counts=True)
    print(f"\n{name}.pt")
    print(" Images shape :", tuple(d["images"].shape))
    print(" Labels shape :", tuple(d["labels"].shape))
    print(" Per class    :", {class_names[i]: int(n) for i, n in zip(u, c)})
    print(f" File size    : {os.path.getsize(f)/1024/1024:.2f} MB")


if __name__ == "__main__":
    paths, labels = collect_paths()
    print("Total images:", len(paths))
    tr_p, va_p, tr_l, va_l = split_data(paths, labels)
    print("Train originals:", len(tr_p), "| Val:", len(va_p))
    build(tr_p, tr_l, va_p, va_l)
    report("train"); report("val")
