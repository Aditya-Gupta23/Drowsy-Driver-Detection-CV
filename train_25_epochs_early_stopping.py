from pathlib import Path
from copy import deepcopy
import json
import time

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, f1_score, accuracy_score


# ---------- Paths ----------
DATA_DIR = Path(
    "/Users/adityagupta/Desktop/Coding/jbooks/HeavyDrivers/archive"
)

# Saves inside HeavyDrivers/model_artifacts when run from the project directory.
OUTPUT_DIR = Path(__file__).resolve().parent / "model_artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = OUTPUT_DIR / "drowsiness_model_25ep_best.pth"
CONFIG_PATH = OUTPUT_DIR / "config_25ep.json"
HISTORY_PATH = OUTPUT_DIR / "training_history_25ep.json"


# ---------- Training configuration ----------
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print("Device:", device)

classes = ["active", "fatigue"]
class_to_idx = {"active": 0, "fatigue": 1}

IMG_SIZE = 224
BATCH_SIZE = 16
MAX_EPOCHS = 25

EARLY_STOPPING_PATIENCE = 5
MIN_F1_IMPROVEMENT = 0.001

INITIAL_LR = 1e-4
WEIGHT_DECAY = 1e-4
LR_PATIENCE = 2
LR_FACTOR = 0.5
MIN_LR = 1e-6

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Reproducibility for model initialization and data shuffling.
SEED = 42
torch.manual_seed(SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)


# ---------- Transforms ----------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(8),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


# ---------- Dataset ----------
class ListedImageDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        self.transform = transform
        self.samples = []

        list_path = DATA_DIR / txt_file
        if not list_path.is_file():
            raise FileNotFoundError(f"Dataset list not found: {list_path}")

        with open(list_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                relative_path, label = parts[0], int(parts[-1])
                path = DATA_DIR / relative_path

                if not path.is_file():
                    raise FileNotFoundError(path)

                if label not in (0, 1):
                    raise ValueError(
                        f"Unexpected label {label} in {list_path}: {line.strip()}"
                    )

                self.samples.append((path, label))

        if not self.samples:
            raise ValueError(f"No samples found in {list_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        with Image.open(path) as im:
            image = im.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


train_ds = ListedImageDataset("train.txt", train_transform)
val_ds = ListedImageDataset("val.txt", eval_transform)
test_ds = ListedImageDataset("test.txt", eval_transform)

for name, ds in [
    ("train", train_ds),
    ("val", val_ds),
    ("test", test_ds),
]:
    counts = {
        c: sum(label == idx for _, label in ds.samples)
        for c, idx in class_to_idx.items()
    }
    print(f"{name}: {len(ds)} images | {counts}")

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)


# ---------- Model ----------
weights = models.MobileNet_V3_Small_Weights.DEFAULT
model = models.mobilenet_v3_small(weights=weights)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=LR_FACTOR,
    patience=LR_PATIENCE,
    threshold=MIN_F1_IMPROVEMENT,
    threshold_mode="abs",
    min_lr=MIN_LR,
)


# ---------- Evaluation helper ----------
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    labels_all, preds_all = [], []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels_device = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels_device)
            total_loss += loss.item() * labels.size(0)

            preds_all.extend(logits.argmax(dim=1).cpu().tolist())
            labels_all.extend(labels.tolist())

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(
        labels_all, preds_all, average="macro", zero_division=0
    )
    accuracy = accuracy_score(labels_all, preds_all)

    return avg_loss, macro_f1, accuracy, labels_all, preds_all


# ---------- Training with early stopping ----------
best_f1 = -1.0
best_epoch = 0
epochs_without_improvement = 0
best_state = None
history = []

for epoch in range(1, MAX_EPOCHS + 1):
    epoch_start = time.time()
    model.train()

    running_loss = 0.0
    train_preds, train_labels = [], []

    for images, labels in train_loader:
        images = images.to(device)
        labels_device = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels_device)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        train_preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
        train_labels.extend(labels.tolist())

    train_loss = running_loss / len(train_ds)
    train_acc = accuracy_score(train_labels, train_preds)

    val_loss, val_f1, val_acc, _, _ = evaluate(
        model, val_loader, criterion
    )

    previous_lr = optimizer.param_groups[0]["lr"]
    scheduler.step(val_f1)
    current_lr = optimizer.param_groups[0]["lr"]

    improved = val_f1 > best_f1 + MIN_F1_IMPROVEMENT

    if improved:
        best_f1 = val_f1
        best_epoch = epoch
        epochs_without_improvement = 0

        # Keep an independent CPU copy of the best weights.
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

        # Save immediately, so the best checkpoint survives interruption.
        torch.save(best_state, BEST_MODEL_PATH)
        print(f"  Saved new best checkpoint: {BEST_MODEL_PATH.name}")
    else:
        epochs_without_improvement += 1

    record = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "validation_loss": val_loss,
        "validation_accuracy": val_acc,
        "validation_macro_f1": val_f1,
        "learning_rate": current_lr,
        "improved": improved,
        "seconds": round(time.time() - epoch_start, 2),
    }
    history.append(record)

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    print(
        f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
        f"train_loss={train_loss:.4f} | "
        f"train_acc={train_acc:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"val_acc={val_acc:.4f} | "
        f"val_macro_f1={val_f1:.4f} | "
        f"lr={current_lr:.2e} | "
        f"no_improve={epochs_without_improvement}/{EARLY_STOPPING_PATIENCE} | "
        f"time={record['seconds']:.1f}s"
    )

    if current_lr < previous_lr:
        print(f"  Learning rate reduced: {previous_lr:.2e} -> {current_lr:.2e}")

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print(
            f"\nEarly stopping at epoch {epoch}. "
            f"Best validation macro-F1 was {best_f1:.4f} at epoch {best_epoch}."
        )
        break


if best_state is None:
    raise RuntimeError("Training ended without saving a best checkpoint.")

# ---------- Load best checkpoint and save matching config ----------
best_state = torch.load(
    BEST_MODEL_PATH, map_location="cpu", weights_only=True
)
model.load_state_dict(best_state)
model = model.to(device)
model.eval()

config = {
    "architecture": "mobilenet_v3_small",
    "classes": classes,
    "class_to_idx": class_to_idx,
    "image_size": IMG_SIZE,
    "mean": mean,
    "std": std,
    "best_validation_macro_f1": best_f1,
    "best_epoch": best_epoch,
    "max_epochs": MAX_EPOCHS,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "checkpoint": BEST_MODEL_PATH.name,
}

with open(CONFIG_PATH, "w") as f:
    json.dump(config, f, indent=2)

# ---------- Final test evaluation ----------
# Use the test set only after selecting the best checkpoint by validation F1.
test_loss, test_f1, test_acc, test_labels, test_preds = evaluate(
    model, test_loader, criterion
)

print("\nBest checkpoint:")
print("Epoch:", best_epoch)
print(f"Validation macro-F1: {best_f1:.4f}")
print("\nFinal held-out test results:")
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test macro-F1: {test_f1:.4f}")
print(classification_report(
    test_labels,
    test_preds,
    labels=[0, 1],
    target_names=classes,
    zero_division=0
))

print("\nSaved files:")
print("Best weights:", BEST_MODEL_PATH.resolve())
print("Matching config:", CONFIG_PATH.resolve())
print("Training history:", HISTORY_PATH.resolve())
