import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms


class FaceAugmenter:
    def __init__(self, size=224):
        self.size = size
        base = [transforms.Resize((size, size)), transforms.ToTensor()]  # tensor first
        self.augs = {
            "h_flip":      transforms.Compose(base + [transforms.RandomHorizontalFlip(p=1.0)]),
            "rotate":      transforms.Compose(base + [transforms.RandomRotation(15)]),
            "brightness":  transforms.Compose(base + [transforms.ColorJitter(brightness=0.4)]),
            "contrast":    transforms.Compose(base + [transforms.ColorJitter(contrast=0.4)]),
            "blur":        transforms.Compose(base + [transforms.GaussianBlur(5, sigma=(0.5, 2.0))]),
            "perspective": transforms.Compose(base + [transforms.RandomPerspective(0.3, p=1.0)]),
        }
        self.names = list(self.augs.keys())
        self.base = transforms.Compose(base)

    def apply_random(self, img):
        name = random.choice(self.names)
        return self.augs[name](img)

    def get_n_augmentations(self, img_path, n):
        img = Image.open(img_path).convert("RGB")
        return [self.apply_random(img) for _ in range(n)]

    def visualize(self, img_path, save_path="augmentation_grid.png"):
        img = Image.open(img_path).convert("RGB")
        items = [("original", self.base(img))]
        for name in self.names:
            items.append((name, self.augs[name](img)))
        cols = 4
        rows = (len(items) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = axes.flatten()
        for ax in axes:
            ax.axis("off")
        for ax, (name, t) in zip(axes, items):
            ax.imshow(t.permute(1, 2, 0).clamp(0, 1).numpy())
            ax.set_title(name)
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        print('Saved', save_path)


if __name__ == "__main__":
    import sys
    FaceAugmenter().visualize(sys.argv[1])
