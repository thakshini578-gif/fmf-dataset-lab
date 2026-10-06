import time
import torch
import torch.nn as nn
from torchvision import models
from loader import create_dataloaders

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)


def run(bs, epochs=5):
    tl, vl, names = create_dataloaders("./fmfdataset", bs)
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.last_channel, len(names))
    m = m.to(device)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    lossf = nn.CrossEntropyLoss()
    losses = []
    t0 = time.time()
    for ep in range(epochs):
        m.train()
        tot = 0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            l = lossf(m(x), y)
            l.backward()
            opt.step()
            tot += l.item()
        losses.append(tot / len(tl))
    m.eval()
    c = n = 0
    with torch.no_grad():
        for x, y in vl:
            x, y = x.to(device), y.to(device)
            c += (m(x).argmax(1) == y).sum().item()
            n += len(y)
    mem = f"{torch.cuda.max_memory_allocated()/1e6:.0f} MB" if device == "cuda" else "n/a (CPU)"
    print(f"batch={bs}: time={time.time()-t0:.1f}s | val_acc={c/n:.2f} | "
          f"loss per epoch={[round(v, 3) for v in losses]} | gpu_mem={mem}")


run(8)
run(64)
