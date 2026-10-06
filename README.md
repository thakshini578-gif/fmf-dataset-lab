# Find My Friend Dataset (Lab 2, Deep Learning, HND Data Science 25.2f)

Scripts for building a face dataset (3 friends + unknown) and loading it with PyTorch.
No face images or .pt files are stored in this repository (privacy).

## Run order
1. setup_dataset.py - creates folders and config.json
2. capture.py / import_existing.py - face detection (Haar cascade) and 20% padding crop
3. augment.py - FaceAugmenter class and augmentation grid
4. build_dataset.py - stratified 80/20 split, augmentation, builds train.pt and val.pt
5. eda.py - class counts, bar chart, sample grid
6. loader.py - FMFDataset and create_dataloaders()
7. compare_batch.py - MobileNet batch size 8 vs 64
