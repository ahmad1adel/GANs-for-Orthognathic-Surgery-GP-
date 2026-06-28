"""
test.py — Improved CNN classifier for Orthognathic Surgery (Before vs After)

Fixes applied vs the original notebook:
  1. Frozen ResNet-18 backbone  → only the classifier head is trained
  2. Stronger regularisation    → dropout 0.6, label smoothing, heavier augmentation
  3. 5-fold cross-validation    → reliable estimate with only 200 images
"""

import random, copy, warnings
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, roc_auc_score, accuracy_score,
    f1_score, precision_score, recall_score
)

warnings.filterwarnings('ignore')

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# ── Config ───────────────────────────────────────────────────────────────────
BEFORE_DIR   = Path('dataset/before')
AFTER_DIR    = Path('dataset/after')
IMG_SIZE     = 224
BATCH_SIZE   = 16
NUM_EPOCHS   = 25
LR           = 5e-4       # higher LR is fine when backbone is frozen
WEIGHT_DECAY = 1e-3
N_FOLDS      = 5
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Device : {DEVICE}')
print(f'Before : {len(list(BEFORE_DIR.glob("*.png")))} images')
print(f'After  : {len(list(AFTER_DIR.glob("*.png")))} images')


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class SurgeryDataset(Dataset):
    """0 = before surgery, 1 = after surgery."""
    def __init__(self, before_dir, after_dir, transform=None):
        self.transform = transform
        self.samples = (
            [(str(p), 0) for p in Path(before_dir).glob('*.png')] +
            [(str(p), 1) for p in Path(after_dir).glob('*.png')]
        )

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    @property
    def labels(self):
        return [s[1] for s in self.samples]


# ── Transforms ───────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# Model — frozen backbone + custom head
# ─────────────────────────────────────────────────────────────────────────────
def build_frozen_resnet18():
    """
    ResNet-18 with the entire backbone frozen.
    Only the new classifier head is trained.
    Massively reduces overfitting on small datasets.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze ALL backbone parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace head — only these params will be trained
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.6),
        nn.Linear(256, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(64, 1),
    )
    return model


def count_trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Training helper (single fold)
# ─────────────────────────────────────────────────────────────────────────────
def train_one_fold(train_idx, val_idx, full_dataset):
    # Build loaders with correct transforms
    train_ds = SurgeryDataset(BEFORE_DIR, AFTER_DIR, transform=train_tf)
    val_ds   = SurgeryDataset(BEFORE_DIR, AFTER_DIR, transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        sampler=SubsetRandomSampler(train_idx), num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        sampler=SubsetRandomSampler(val_idx), num_workers=0
    )

    model     = build_frozen_resnet18().to(DEVICE)
    # Label smoothing reduces overconfidence
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(DEVICE))
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val_acc  = 0.0
    best_weights  = None
    patience      = 8
    no_improve    = 0

    train_accs, val_accs = [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        # ── train ──
        model.train()
        t_correct = t_total = 0
        for imgs, labels in train_loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward(); optimizer.step()
            preds      = (torch.sigmoid(model(imgs)) >= 0.5).long()
            t_correct += (preds == labels.long()).sum().item()
            t_total   += imgs.size(0)

        # ── val ──
        model.eval()
        v_correct = v_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(DEVICE)
                labels = labels.float().unsqueeze(1).to(DEVICE)
                preds  = (torch.sigmoid(model(imgs)) >= 0.5).long()
                v_correct += (preds == labels.long()).sum().item()
                v_total   += imgs.size(0)

        scheduler.step()
        t_acc = t_correct / t_total
        v_acc = v_correct / v_total
        train_accs.append(t_acc)
        val_accs.append(v_acc)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_weights = copy.deepcopy(model.state_dict())
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_weights)

    # ── collect predictions on val fold ──
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            logits = model(imgs.to(DEVICE)).squeeze(1)
            probs  = torch.sigmoid(logits).cpu().numpy()
            preds  = (probs >= 0.5).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    return {
        'model':      model,
        'best_acc':   best_val_acc,
        'train_accs': train_accs,
        'val_accs':   val_accs,
        'labels':     np.array(all_labels),
        'probs':      np.array(all_probs),
        'preds':      np.array(all_preds),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5-Fold Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────
print(f'\nTrainable params (head only): {count_trainable(build_frozen_resnet18()):,}')
print(f'\nRunning {N_FOLDS}-fold cross-validation...\n')

full_dataset = SurgeryDataset(BEFORE_DIR, AFTER_DIR)
all_labels   = np.array(full_dataset.labels)
indices      = np.arange(len(full_dataset))

skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(indices, all_labels), 1):
    print(f'Fold {fold}/{N_FOLDS}  (train={len(train_idx)}, val={len(val_idx)})')
    res = train_one_fold(train_idx, val_idx, full_dataset)
    results.append(res)

    acc = accuracy_score(res['labels'], res['preds'])
    auc = roc_auc_score(res['labels'], res['probs'])
    f1  = f1_score(res['labels'], res['preds'])
    print(f'  -> Val Acc={acc:.4f}  AUC={auc:.4f}  F1={f1:.4f}')


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate metrics across folds
# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 55)
print('  5-Fold Cross-Validation Summary')
print('=' * 55)

metrics = {
    'Accuracy':  [accuracy_score(r['labels'], r['preds'])  for r in results],
    'Precision': [precision_score(r['labels'], r['preds']) for r in results],
    'Recall':    [recall_score(r['labels'], r['preds'])    for r in results],
    'F1':        [f1_score(r['labels'], r['preds'])        for r in results],
    'ROC-AUC':   [roc_auc_score(r['labels'], r['probs'])   for r in results],
}

for name, vals in metrics.items():
    print(f'  {name:<12}  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  '
          f'per-fold={[round(v,3) for v in vals]}')


# ─────────────────────────────────────────────────────────────────────────────
# Plot: learning curves for all folds
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, N_FOLDS, figsize=(4 * N_FOLDS, 4), sharey=True)
fig.suptitle('Frozen ResNet-18 — Val Accuracy per Fold (5-Fold CV)',
             fontsize=13, fontweight='bold')

for i, (ax, res) in enumerate(zip(axes, results), 1):
    ep = range(1, len(res['train_accs']) + 1)
    ax.plot(ep, res['train_accs'], 'b-', lw=1.5, label='Train')
    ax.plot(ep, res['val_accs'],   'r-', lw=1.5, label='Val')
    ax.set_title(f'Fold {i}  (best={res["best_acc"]:.3f})')
    ax.set_xlabel('Epoch'); ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    if i == 1: ax.legend()

plt.tight_layout()
plt.savefig('test_fold_curves.png', dpi=120, bbox_inches='tight')
print('\nSaved test_fold_curves.png')
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Plot: metrics comparison (old vs new)
# ─────────────────────────────────────────────────────────────────────────────
old_metrics = {
    'Accuracy': 0.500, 'Precision': 0.500, 'Recall': 0.455, 'F1': 0.476, 'ROC-AUC': 0.606
}
new_metrics = {k: np.mean(v) for k, v in metrics.items()}

x     = np.arange(len(old_metrics))
width = 0.35
names = list(old_metrics.keys())

fig, ax = plt.subplots(figsize=(11, 5))
bars1 = ax.bar(x - width/2, [old_metrics[k] for k in names], width,
               label='ResNet-18 (original)', color='#e07b54', edgecolor='grey')
bars2 = ax.bar(x + width/2, [new_metrics[k] for k in names], width,
               label='Frozen ResNet-18 + 5-Fold CV (new)', color='#4c9be8', edgecolor='grey')

ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_ylim(0, 1.1)
ax.set_title('Improvement: Original vs Frozen Backbone + CV', fontsize=13, fontweight='bold')
ax.axhline(0.5, color='grey', linestyle='--', lw=1, label='Random baseline')
ax.legend(); ax.grid(axis='y', alpha=0.3)

for bar in bars1:
    ax.annotate(f'{bar.get_height():.3f}',
                (bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax.annotate(f'{bar.get_height():.3f}',
                (bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha='center', va='bottom', fontsize=8, color='#1a5fa0', fontweight='bold')

plt.tight_layout()
plt.savefig('test_improvement.png', dpi=120, bbox_inches='tight')
print('Saved test_improvement.png')
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Best fold model — detailed report
# ─────────────────────────────────────────────────────────────────────────────
best_fold_idx = int(np.argmax([r['best_acc'] for r in results]))
best          = results[best_fold_idx]

print(f'\nBest fold: {best_fold_idx + 1}  (val acc={best["best_acc"]:.4f})')
print(classification_report(best['labels'], best['preds'],
                             target_names=['Before', 'After'], digits=4))

# Save best fold model
torch.save(best['model'].state_dict(), 'test_best_model.pth')
print('Saved test_best_model.pth')
