import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class FMFDataset(Dataset):
    def __init__(self, pt_file, transform=None):
        d = torch.load(pt_file, weights_only=False)
        self.images = d["images"]
        self.labels = d["labels"]
        self.paths = d["paths"]
        self.class_names = d["class_names"]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.images[i]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]


def create_dataloaders(root, batch_size=8):
    norm = transforms.Normalize(MEAN, STD)
    tr = FMFDataset(os.path.join(root, "processed", "train.pt"), norm)
    va = FMFDataset(os.path.join(root, "processed", "val.pt"), norm)
    return (DataLoader(tr, batch_size=batch_size, shuffle=True),
            DataLoader(va, batch_size=batch_size, shuffle=False),
            tr.class_names)


if __name__ == "__main__":
    train_loader, val_loader, class_names = create_dataloaders("./fmfdataset", batch_size=8)
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")
    print(f"Label names: {[class_names[l] for l in labels]}")
