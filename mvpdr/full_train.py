import os
import sys
import json
import time
import random
import logging
import argparse
from pathlib import Path
from collections import Counter

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / 'data' / 'plantwild' / 'plantwild'
OUT = PROJECT_ROOT / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'logs').mkdir(exist_ok=True)
(OUT / 'checkpoints').mkdir(exist_ok=True)
(OUT / 'graphs').mkdir(exist_ok=True)
MODELS_DIR = PROJECT_ROOT / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger('terramind_train')
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(ch)
fh = logging.FileHandler(OUT / 'logs' / 'full_training.log', mode='w')
fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.addHandler(fh)

class BatchProgress:
    def __init__(self, num_batches, log_every=25, label='Batch'):
        self.num_batches = max(1, num_batches)
        self.log_every = max(1, log_every)
        self.label = label
        self.start = time.time()
        self.ma = 0.0
    def step(self, bi, extra=''):
        if (bi + 1) % self.log_every == 0 or (bi + 1) == self.num_batches:
            elapsed = time.time() - self.start
            rate = (bi + 1) / max(elapsed, 1e-6)
            eta = (self.num_batches - bi - 1) / max(rate, 1e-6)
            pct = 100.0 * (bi + 1) / self.num_batches
            msg = (f'  {self.label} {bi+1:>4}/{self.num_batches}  '
                   f'[{pct:5.1f}%]  {rate:.2f}/s  '
                   f'elapsed={elapsed:>6.1f}s  eta~{eta:>5.0f}s')
            if extra:
                msg += f'  {extra}'
            logger.info(msg)
            sys.stdout.flush()
            try:
                sys.stderr.flush()
            except Exception:
                pass

parser = argparse.ArgumentParser(description='Full PlantWild training (CPU-optimized, MobileNetV3-Small, batch progress).')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=22)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--early-stop-min-delta', type=float, default=0.005,
                    help='Minimum validation-loss reduction required to reset early stopping')
parser.add_argument('--img-size', type=int, default=160, help='Spatial image size (default 160 for CPU speed)')
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--accum', type=int, default=4, help='Gradient accumulation steps (eff batch = batch*accum)')
parser.add_argument('--dropout', type=float, default=0.45)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--lr', type=float, default=4e-4)
parser.add_argument('--log-every', type=int, default=40, help='Batch progress log cadence')
parser.add_argument('--limit-train', type=int, default=0, help='0 = all (12,979) train samples')
parser.add_argument('--pretrained', type=int, default=1, help='1=use pretrained MobileNetV3-Small weights')
args = parser.parse_args()

SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

IMG = args.img_size
BATCH = args.batch_size
ACCUM = max(1, args.accum)
EFF_BATCH = BATCH * ACCUM
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.info('=' * 70)
logger.info('TerraMind AI — Full PlantWild Training Pipeline (CPU-Friendly)')
logger.info('=' * 70)
logger.info(f'Device: {device}  Torch: {torch.__version__}')
logger.info(f'Img={IMG}px  Batch={BATCH}  Accum={ACCUM}  EffBatch={EFF_BATCH}')
logger.info(f'Epochs={args.epochs}  Patience={args.patience}  Seed={SEED}')
logger.info(f'LR={args.lr}  WD={args.weight_decay}  Dropout={args.dropout}')

with open(ROOT / 'classes.txt', 'r', encoding='utf-8') as f:
    classes_map = {}
    for line in f:
        line = line.strip()
        if not line:
            continue
        idx, name = line.split(' ', 1)
        classes_map[name] = int(idx)
num_classes = len(classes_map)
idx_to_class = {v: k for k, v in classes_map.items()}
logger.info(f'Classes: {num_classes}')

with open(ROOT / 'trainval.txt', 'r', encoding='utf-8') as f:
    all_lines = [l.strip() for l in f if l.strip()]
logger.info(f'Dataset entries in trainval.txt: {len(all_lines)}')

class PlantWildDataset(Dataset):
    def __init__(self, root, split_lines, classes_map, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.items = []
        self.classes_map = classes_map
        missing = 0
        for ln in split_lines:
            if not ln.strip():
                continue
            parts = ln.strip().split('=')
            rel = parts[0]
            if not Path(rel).parts[0].lower().startswith('images'):
                rel = 'images/' + rel
            imgp = self.root / rel
            cls_name = Path(parts[0]).parts[0]
            label = classes_map.get(cls_name)
            if label is None:
                continue
            if not imgp.is_file():
                missing += 1
                continue
            self.items.append((imgp, label))
        if missing:
            logger.warning(f'Skipped {missing} missing images from {len(split_lines)} entries')
        if transform is None:
            self.transform = T.Compose([
                T.Resize((IMG, IMG)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        p, label = self.items[idx]
        try:
            with Image.open(p) as source:
                img = source.convert('RGB')
            img = self.transform(img)
            return img, label
        except Exception:
            return torch.zeros(3, IMG, IMG), 0

def official_split(lines):
    """Use PlantWild's supplied mode labels: 1=train, 2=validation, 0=test."""
    splits = {'0': [], '1': [], '2': []}
    for line in lines:
        try:
            _, _, mode = line.rsplit('=', 2)
        except ValueError as exc:
            raise ValueError(f'Malformed split entry: {line!r}') from exc
        if mode not in splits:
            raise ValueError(f'Unknown PlantWild split mode {mode!r}: {line!r}')
        splits[mode].append(line)
    return splits['1'], splits['2'], splits['0']

def assert_disjoint_splits(train_l, val_l, test_l):
    def paths(lines):
        return {line.split('=', 1)[0] for line in lines}
    train_paths, val_paths, test_paths = paths(train_l), paths(val_l), paths(test_l)
    overlaps = {
        'train/validation': train_paths & val_paths,
        'train/test': train_paths & test_paths,
        'validation/test': val_paths & test_paths,
    }
    leaked = {name: values for name, values in overlaps.items() if values}
    if leaked:
        details = ', '.join(f'{name}={len(values)}' for name, values in leaked.items())
        raise RuntimeError(f'Dataset split leakage detected: {details}')

train_lines, val_lines, test_lines = official_split(all_lines)
assert_disjoint_splits(train_lines, val_lines, test_lines)
logger.info(f'Official PlantWild split: Train={len(train_lines)}, Val={len(val_lines)}, Test={len(test_lines)}')

split_info = {'seed': SEED, 'train_count': len(train_lines), 'val_count': len(val_lines),
              'test_count': len(test_lines), 'source': 'PlantWild supplied mode labels (1=train, 2=val, 0=test)'}
with open(OUT / 'split_info.json', 'w') as f:
    json.dump(split_info, f, indent=2)

for fn, lst in [('train_split.txt', train_lines), ('val_split.txt', val_lines), ('test_split.txt', test_lines)]:
    with open(ROOT / fn, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lst) + '\n')

def class_stats(lines, label):
    c = Counter(ln.split('=')[0].split('/')[0] for ln in lines)
    s = list(c.values())
    logger.info(f'{label}: n={len(lines)} classes={len(c)} min={min(s)} max={max(s)} mean={np.mean(s):.1f}')

def stratified_subset(lines, limit, seed):
    """Create a deterministic tuning subset that represents every class."""
    if limit >= len(lines):
        return list(lines)
    rng = random.Random(seed)
    by_class = {}
    for line in lines:
        by_class.setdefault(line.split('=', 1)[0].split('/')[0], []).append(line)
    selected = []
    pool = []
    for items in by_class.values():
        rng.shuffle(items)
        selected.append(items[0])
        pool.extend(items[1:])
    rng.shuffle(pool)
    selected.extend(pool[:max(0, limit - len(selected))])
    rng.shuffle(selected)
    return selected[:limit]

class_stats(train_lines, 'Train')
class_stats(val_lines, 'Val  ')
class_stats(test_lines, 'Test ')

if args.limit_train > 0:
    def stratified_limit(lines, limit):
        grouped = {}
        for ln in lines:
            grouped.setdefault(ln.split('=')[0].split('/')[0], []).append(ln)
        selected = [items[0] for items in grouped.values()]
        pool = [ln for items in grouped.values() for ln in items[1:]]
        random.Random(SEED).shuffle(pool)
        selected.extend(pool[:max(0, limit - len(selected))])
        random.Random(SEED).shuffle(selected)
        return selected
    train_lines = stratified_limit(train_lines, min(args.limit_train, len(train_lines)))
    logger.warning(f'LIMITED train set to {len(train_lines)} samples (use --limit-train=0 to train on all 12,979)')

train_transform = T.Compose([
    T.Resize((int(IMG * 1.15), int(IMG * 1.15))),
    T.RandomCrop((IMG, IMG)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=18),
    T.ColorJitter(brightness=0.22, contrast=0.22, saturation=0.18, hue=0.04),
    T.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08)),
    T.ToTensor(),
    T.RandomErasing(p=0.22, scale=(0.02, 0.18), ratio=(0.3, 3.3), value='random'),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_test_transform = T.Compose([
    T.Resize((int(IMG * 1.15), int(IMG * 1.15))),
    T.CenterCrop((IMG, IMG)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def build_model(dropout=0.45, pretrained=True):
    try:
        if pretrained:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            m = models.mobilenet_v3_small(weights=weights)
            logger.info('[build_model] Using pretrained MobileNetV3-Small weights')
        else:
            m = models.mobilenet_v3_small(weights=None)
    except Exception as e:
        logger.warning(f'MobileNetV3-Small pretrained failed: {e}; using ResNet18')
        if pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
                m = models.resnet18(weights=weights)
            except Exception:
                m = models.resnet18(weights=None)
        in_feats = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_feats, 384), nn.ReLU(), nn.BatchNorm1d(384),
            nn.Dropout(p=dropout * 0.6),
            nn.Linear(384, num_classes),
        )
        return m.to(device)
    in_feats = m.classifier[0].in_features
    m.classifier = nn.Sequential(
        nn.Linear(in_feats, 576), nn.Hardswish(inplace=True),
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(576, 256), nn.ReLU(), nn.BatchNorm1d(256),
        nn.Dropout(p=dropout * 0.55),
        nn.Linear(256, num_classes),
    )
    return m.to(device)

def freeze_backbone(model, freeze=True):
    backbone = model.features if hasattr(model, 'features') else (list(model.children())[:-1])
    if hasattr(model, 'features'):
        for p in model.features.parameters():
            p.requires_grad = not freeze
    else:
        modules = list(model.children())[:-1]
        for mod in modules:
            for p in mod.parameters():
                p.requires_grad = not freeze
    for p in model.classifier.parameters():
        p.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'[freeze] Backbone frozen={freeze}; total params={total:,}; trainable={trainable:,}')
    return total, trainable

def train_one_epoch(model, dl, opt, crit, device, accum=1, grad_clip=1.0, log_every=40):
    model.train()
    total = 0
    correct = 0
    loss_sum = 0.0
    n_batches = len(dl)
    bp = BatchProgress(n_batches, log_every=log_every, label='TrainBatch')
    opt.zero_grad(set_to_none=True)
    for bi, (imgs, labels) in enumerate(dl):
        imgs = imgs.to(device)
        labels = torch.as_tensor(labels).to(device)
        out = model(imgs)
        loss = crit(out, labels) / accum
        loss.backward()
        if (bi + 1) % accum == 0 or (bi + 1) == n_batches:
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            real_loss = loss.item() * accum
            loss_sum += real_loss * imgs.size(0)
            preds = out.argmax(1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            if (bi + 1) % bp.log_every == 0 or (bi + 1) == n_batches:
                bl = real_loss
                ba = (preds == labels).float().mean().item()
                run_l = loss_sum / max(total, 1)
                run_a = correct / max(total, 1)
                bp.step(bi, extra=f'batch_loss={bl:.3f} batch_acc={ba:.3f} run_loss={run_l:.3f} run_acc={run_a:.3f}')
    return loss_sum / max(total, 1), correct / max(total, 1)

@torch.no_grad()
def evaluate(model, dl, crit, device, log_label='Val  ', log_every=30):
    model.eval()
    total = 0; correct = 0; loss_sum = 0.0
    all_preds, all_labels, all_probs = [], [], []
    n_batches = len(dl)
    bp = BatchProgress(n_batches, log_every=log_every, label=f'{log_label.strip()}Batch')
    for bi, (imgs, labels) in enumerate(dl):
        imgs = imgs.to(device)
        labels = torch.as_tensor(labels).to(device)
        out = model(imgs)
        loss = crit(out, labels)
        loss_sum += loss.item() * imgs.size(0)
        probs = torch.softmax(out, dim=1)
        preds = out.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        if (bi + 1) % bp.log_every == 0 or (bi + 1) == n_batches:
            bp.step(bi, extra=f'run_loss={loss_sum/max(total,1):.3f} run_acc={correct/max(total,1):.3f}')
    avg_loss = loss_sum / max(total, 1)
    avg_acc = correct / max(total, 1)
    return avg_loss, avg_acc, all_preds, all_labels, all_probs

# ================= STAGE 1: Quick warm-up / sanity-tune =================
logger.info('=' * 70)
logger.info('STAGE 1: Warm-up & sanity tuning (short runs, 2 configs)')
logger.info('=' * 70)

warm_train = stratified_subset(train_lines, min(len(train_lines), 1500), seed=SEED)
warm_val = stratified_subset(val_lines, min(len(val_lines), 800), seed=SEED + 1)

warm_train_ds = PlantWildDataset(ROOT, warm_train, classes_map, transform=train_transform)
warm_val_ds = PlantWildDataset(ROOT, warm_val, classes_map, transform=val_test_transform)
warm_train_dl = DataLoader(warm_train_ds, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=False)
warm_val_dl = DataLoader(warm_val_ds, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=False)

tune_cfgs = [
    {'lr': args.lr, 'wd': args.weight_decay, 'dropout': args.dropout, 'desc': f'mbv3s lr={args.lr:.0e} wd={args.weight_decay:.0e} d={args.dropout}'},
    {'lr': args.lr * 0.5, 'wd': args.weight_decay * 2, 'dropout': args.dropout + 0.1, 'desc': f'mbv3s lr={args.lr*0.5:.0e} wd={args.weight_decay*2:.0e} d={args.dropout+0.1}'},
]
tune_results = []
best_warm_val = -1.0
best_warm_idx = 0

for ci, cfg in enumerate(tune_cfgs):
    logger.info(f'--- Tune config {ci+1}/{len(tune_cfgs)}: {cfg["desc"]} ---')
    torch.manual_seed(SEED + ci)
    model = build_model(dropout=cfg['dropout'], pretrained=bool(args.pretrained))
    freeze_backbone(model, freeze=True)
    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg['lr'], weight_decay=cfg['wd'])
    crit = nn.CrossEntropyLoss(label_smoothing=0.08)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2, eta_min=cfg['lr'] * 0.05)
    hist = []
    for ep in range(1, 3):
        logger.info(f'  Tuning epoch {ep}/2')
        t0 = time.time()
        tr_l, tr_a = train_one_epoch(model, warm_train_dl, opt, crit, device, accum=1, grad_clip=1.0, log_every=args.log_every)
        va_l, va_a, *_ = evaluate(model, warm_val_dl, crit, device, log_label='Val  ', log_every=max(1, args.log_every // 2))
        sched.step()
        hist.append({'ep': ep, 'tr_l': tr_l, 'tr_a': tr_a, 'va_l': va_l, 'va_a': va_a})
        logger.info(f'  Warm ep{ep}: tr_acc={tr_a:.3f} va_acc={va_a:.3f} tr_loss={tr_l:.3f} va_loss={va_l:.3f} time={time.time()-t0:.0f}s')
    best_va = max(h['va_a'] for h in hist)
    tune_results.append({'cfg': cfg, 'best_va': best_va, 'hist': hist})
    if best_va > best_warm_val:
        best_warm_val = best_va
        best_warm_idx = ci
    logger.info(f'  -> best_val_acc={best_va:.4f}')

logger.info(f'Warm-up tuning done. Best config #{best_warm_idx+1}: {tune_results[best_warm_idx]["cfg"]["desc"]}  val_acc={best_warm_val:.4f}')
BEST = tune_results[best_warm_idx]['cfg']
with open(OUT / 'tune_results.json', 'w') as f:
    json.dump(tune_results, f, indent=2, default=str)

FINAL = {
    'backbone': 'MobileNetV3-Small' + ('+pretrained' if args.pretrained else '-random-init'),
    'img_size': IMG,
    'batch_size': BATCH,
    'accum_steps': ACCUM,
    'eff_batch': EFF_BATCH,
    'lr': BEST['lr'],
    'weight_decay': BEST['wd'],
    'dropout': BEST['dropout'],
    'grad_clip': 1.0,
    'label_smoothing': 0.10,
    'optimizer': 'AdamW',
    'scheduler': 'CosineAnnealingWarmRestarts(T_0=3, T_mult=2, eta_min=1e-6)',
    'sampler': 'shuffle=True (no replacement oversampling)',
    'gradual_unfreeze_epoch': 4,
    'backbone_lr_after_unfreeze': 1e-5,
    'classifier_lr_after_unfreeze': 1e-4,
    'early_stop_min_delta': args.early_stop_min_delta,
    'augmentation': 'RandomCrop+HorizontalFlip+Rotation+ColorJitter+Affine+RandomErasing',
    'epochs': args.epochs,
    'patience': args.patience,
    'seed': SEED,
}
logger.info(f'Final config: {json.dumps(FINAL, indent=2, default=str)}')
with open(OUT / 'final_config.json', 'w') as f:
    json.dump(FINAL, f, indent=2, default=str)

# ================= STAGE 2: Full Training =================
logger.info('=' * 70)
logger.info(f'STAGE 2: Full Training on ALL {len(train_lines)} train samples')
logger.info('=' * 70)

train_ds = PlantWildDataset(ROOT, train_lines, classes_map, transform=train_transform)
val_ds = PlantWildDataset(ROOT, val_lines, classes_map, transform=val_test_transform)
test_ds = PlantWildDataset(ROOT, test_lines, classes_map, transform=val_test_transform)

train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=False)
val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=False)
test_dl = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=False)

logger.info(f'Loaders: Train ds={len(train_ds)} batches={len(train_dl)} | Val ds={len(val_ds)} batches={len(val_dl)} | Test ds={len(test_ds)} batches={len(test_dl)}')

torch.manual_seed(SEED)
model = build_model(dropout=FINAL['dropout'], pretrained=bool(args.pretrained))
freeze_backbone(model, freeze=True)

crit = nn.CrossEntropyLoss(label_smoothing=FINAL['label_smoothing'])
opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=FINAL['lr'], weight_decay=FINAL['weight_decay'])
sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=3, T_mult=2, eta_min=1e-6)

UNFREEZE_EPOCH = FINAL['gradual_unfreeze_epoch']
history = []
best_val_acc = -1.0
best_val_loss = float('inf')
best_epoch = 0
stale = 0
lrs = []

for epoch in range(1, FINAL['epochs'] + 1):
    ep_start = time.time()

    if epoch == UNFREEZE_EPOCH:
        freeze_backbone(model, freeze=False)
        opt = optim.AdamW([
            {'params': model.features.parameters(), 'lr': FINAL['backbone_lr_after_unfreeze']},
            {'params': model.classifier.parameters(), 'lr': FINAL['classifier_lr_after_unfreeze']},
        ], weight_decay=FINAL['weight_decay'])
        sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=4, T_mult=2, eta_min=5e-7)
        logger.info(f'Epoch {epoch}: *** UNFROZE full backbone; backbone LR={opt.param_groups[0]["lr"]:.1e}, classifier LR={opt.param_groups[1]["lr"]:.1e} ***')

    logger.info(f'--- Epoch {epoch}/{FINAL["epochs"]} (seed={SEED}) ---')
    tr_l, tr_a = train_one_epoch(model, train_dl, opt, crit, device,
                                 accum=ACCUM, grad_clip=FINAL['grad_clip'],
                                 log_every=args.log_every)

    logger.info(f'Validating on {len(val_ds)} samples...')
    va_l, va_a, val_preds, val_labels, _ = evaluate(model, val_dl, crit, device,
                                                     log_label='Val  ', log_every=max(1, args.log_every // 2))

    cur_lr = opt.param_groups[0]['lr']
    lrs.append(cur_lr)
    sched.step()

    elapsed = time.time() - ep_start
    logger.info(f'Epoch {epoch:2d} SUMMARY: '
                f'train_acc={tr_a:.4f}  val_acc={va_a:.4f}  '
                f'train_loss={tr_l:.4f}  val_loss={va_l:.4f}  '
                f'lr={cur_lr:.2e}  total_time={elapsed:.0f}s')

    history.append({'epoch': epoch, 'train_loss': float(tr_l), 'train_acc': float(tr_a),
                    'val_loss': float(va_l), 'val_acc': float(va_a),
                    'lr': float(cur_lr), 'time_s': float(elapsed)})

    try:
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'opt_state': opt.state_dict(),
                    'val_acc': va_a, 'val_loss': va_l}, OUT / 'checkpoints' / f'epoch_{epoch}.pth')
    except Exception as e:
        logger.warning(f'Could not save epoch checkpoint: {e}')

    improved = va_l < best_val_loss - FINAL['early_stop_min_delta']
    if improved:
        best_val_loss = va_l
        best_val_acc = va_a
        best_epoch = epoch
        stale = 0
        try:
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'opt_state': opt.state_dict(),
                        'val_acc': va_a, 'val_loss': va_l,
                        'val_preds': val_preds, 'val_labels': val_labels},
                       OUT / 'checkpoints' / 'best_model.pth')
            logger.info(f'  => NEW BEST saved (val_acc={va_a:.4f}, val_loss={va_l:.4f})')
        except Exception as e:
            logger.warning(f'Could not save best checkpoint: {e}')
    else:
        stale += 1
        logger.info(f'  => No validation-loss improvement >= {FINAL["early_stop_min_delta"]:.4f}. Stale: {stale}/{FINAL["patience"]}  (best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f})')
        if stale >= FINAL['patience']:
            logger.info(f'*** EARLY STOPPING at epoch {epoch} (no improvement for {stale} epochs) ***')
            break

# ================= STAGE 3: Final evaluation on unseen test set =================
logger.info('=' * 70)
logger.info('STAGE 3: Final evaluation — restore BEST model, test on HELD-OUT TEST SET')
logger.info('=' * 70)

best_ckpt = OUT / 'checkpoints' / 'best_model.pth'
if best_ckpt.exists():
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    logger.info(f'Restored BEST checkpoint from epoch {ckpt.get("epoch", best_epoch)}  val_acc={ckpt.get("val_acc", best_val_acc):.4f}')
else:
    logger.warning('No best checkpoint — using last model')

def final_eval_set(label, lines, dl):
    logger.info(f'Evaluating on {label.upper()} set ({len(lines)} samples)...')
    return evaluate(model, dl, crit, device, log_label=f'{label}  ', log_every=max(1, args.log_every // 2))

train_loss_f, train_acc_f, *_ = final_eval_set('train', train_lines, DataLoader(
    PlantWildDataset(ROOT, train_lines, classes_map, transform=val_test_transform),
    batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=False))
val_loss_f, val_acc_f, *_ = final_eval_set('val', val_lines, val_dl)
test_loss, test_acc, test_preds, test_labels, test_probs = final_eval_set('test', test_lines, test_dl)

logger.info(f'TRAIN (eval mode, no aug) : acc={train_acc_f:.4f}  loss={train_loss_f:.4f}')
logger.info(f'VAL   (best checkpoint)    : acc={val_acc_f:.4f}  loss={val_loss_f:.4f}')
logger.info(f'TEST  (unseen holdout)     : acc={test_acc:.4f}  loss={test_loss:.4f}')

# sklearn metrics
try:
    from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
    y_true = np.array(test_labels); y_pred = np.array(test_preds)
    macro_p = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_r = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    report_dict = classification_report(y_true, y_pred, target_names=[idx_to_class[i] for i in range(num_classes)],
                                         zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist()
    per_class = {idx_to_class[i]: report_dict.get(idx_to_class[i], {}) for i in range(num_classes)}
    logger.info(f'TEST Macro: P={macro_p:.4f}  R={macro_r:.4f}  F1={macro_f1:.4f}')
    logger.info(f'TEST Weighted F1={weighted_f1:.4f}')
except Exception as e:
    logger.warning(f'sklearn metrics failed: {e}')
    macro_p = macro_r = macro_f1 = weighted_f1 = 0.0
    per_class = {}
    cm = []

overfit_gap_tv = max(0.0, (train_acc_f - val_acc_f) * 100)
overfit_gap_vt = max(0.0, (val_acc_f - test_acc) * 100)
overfit_gap_tt = max(0.0, (train_acc_f - test_acc) * 100)
if overfit_gap_tt <= 5:
    level = 'LOW (excellent generalization)'
elif overfit_gap_tt <= 15:
    level = 'MODERATE (acceptable, generalization OK)'
elif overfit_gap_tt <= 25:
    level = 'NOTICEABLE (some overfitting, regularization could help)'
else:
    level = 'HIGH (strong overfitting, needs more reg / data)'

logger.info(f'Overfitting Gaps (accuracy percentage points):')
logger.info(f'  Train -> Val  : {overfit_gap_tv:.2f} pp')
logger.info(f'  Val   -> Test : {overfit_gap_vt:.2f} pp')
logger.info(f'  Train -> Test : {overfit_gap_tt:.2f} pp')
logger.info(f'  Generalization: {level}')

# ================= Plots =================
logger.info('Generating training plots...')
epochs_ran = list(range(1, len(history) + 1))
tr_l = [h['train_loss'] for h in history]
va_l = [h['val_loss'] for h in history]
tr_a = [h['train_acc'] for h in history]
va_a = [h['val_acc'] for h in history]

fig, axes = plt.subplots(2, 2, figsize=(15, 11))

ax = axes[0, 0]
ax.plot(epochs_ran, tr_l, 'b-o', ms=3, label='Train Loss')
ax.plot(epochs_ran, va_l, 'r-s', ms=3, label='Val Loss')
ax.axvline(best_epoch, color='g', ls='--', alpha=0.7, label=f'Best epoch {best_epoch}')
ax.set_xlabel('Epoch'); ax.set_ylabel('CE Loss (smoothed)'); ax.set_title('Loss vs Epoch')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(epochs_ran, tr_a, 'b-o', ms=3, label='Train Acc')
ax.plot(epochs_ran, va_a, 'r-s', ms=3, label='Val Acc')
ax.axhline(test_acc, color='m', ls=':', alpha=0.75, label=f'Test Acc = {test_acc:.4f}')
ax.axvline(best_epoch, color='g', ls='--', alpha=0.7, label=f'Best epoch {best_epoch}')
ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy'); ax.set_title('Accuracy vs Epoch')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(epochs_ran, lrs, 'k-', lw=2)
ax.set_xlabel('Epoch'); ax.set_ylabel('Learning Rate'); ax.set_title('LR Schedule (CosineAnnealingWarmRestarts)')
ax.set_yscale('log'); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
split_labels = ['Train (eval)', 'Val', 'Test (unseen)']
split_accs = [train_acc_f, val_acc_f, test_acc]
colors = ['#4C78A8', '#F58518', '#54A24B']
bars = ax.bar(split_labels, split_accs, color=colors, edgecolor='black')
for bar, v in zip(bars, split_accs):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f'{v:.4f}', ha='center', fontweight='bold')
ax.set_ylabel('Accuracy'); ax.set_title('Final Performance Across Splits')
ax.set_ylim(0, min(1.0, max(split_accs) * 1.2 + 0.05))
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle(f'TerraMind AI — {FINAL["backbone"]} @ {IMG}px\n'
             f'Best Val Acc={best_val_acc:.4f}  |  Test Acc={test_acc:.4f}  |  Generalization: {level}',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT / 'graphs' / 'training_curves.png', dpi=150, bbox_inches='tight')
plt.close()

if cm:
    fig, ax = plt.subplots(figsize=(16, 8))
    arr = np.array(cm, dtype=float)
    rs = arr.sum(axis=1, keepdims=True)
    nrm = np.divide(arr, rs, out=np.zeros_like(arr), where=rs != 0)
    im = ax.imshow(nrm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    ax.set_title('Normalized Confusion Matrix (Test Set)')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    if num_classes <= 30:
        ax.set_xticks(range(num_classes))
        ax.set_xticklabels([idx_to_class[i] for i in range(num_classes)], rotation=90, fontsize=7)
        ax.set_yticks(range(num_classes))
        ax.set_yticklabels([idx_to_class[i] for i in range(num_classes)], fontsize=7)
    else:
        step = max(1, num_classes // 20)
        ticks = list(range(0, num_classes, step))
        ax.set_xticks(ticks); ax.set_xticklabels([idx_to_class[i] for i in ticks], rotation=90, fontsize=7)
        ax.set_yticks(ticks); ax.set_yticklabels([idx_to_class[i] for i in ticks], fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(OUT / 'graphs' / 'confusion_matrix.png', dpi=120, bbox_inches='tight')
    plt.close()

# ================= Save artifacts =================
final_model_path = MODELS_DIR / 'final_model.pth'
try:
    torch.save({
        'model_state': model.state_dict(),
        'num_classes': num_classes,
        'classes': classes_map,
        'idx_to_class': idx_to_class,
        'backbone': 'MobileNetV3-Small',
        'img_size': IMG,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'best_val_acc': float(best_val_acc),
        'best_val_loss': float(best_val_loss),
        'best_epoch': best_epoch,
        'test_acc': float(test_acc),
        'test_loss': float(test_loss),
        'test_macro_f1': float(macro_f1),
        'final_config': FINAL,
    }, final_model_path)
    logger.info(f'Final model saved -> {final_model_path}')
except Exception as e:
    logger.warning(f'Could not save final model: {e}')

report = {
    'final_config': FINAL,
    'best_epoch': best_epoch,
    'epochs_ran': len(history),
    'best_val_acc': float(best_val_acc),
    'best_val_loss': float(best_val_loss),
    'train_acc_final_eval': float(train_acc_f),
    'val_acc_final_eval': float(val_acc_f),
    'test_loss': float(test_loss),
    'test_acc': float(test_acc),
    'test_macro_precision': float(macro_p),
    'test_macro_recall': float(macro_r),
    'test_macro_f1': float(macro_f1),
    'test_weighted_f1': float(weighted_f1),
    'overfitting': {
        'train_val_gap_pp': float(overfit_gap_tv),
        'val_test_gap_pp': float(overfit_gap_vt),
        'train_test_gap_pp': float(overfit_gap_tt),
        'assessment': level,
    },
    'per_class_test': per_class,
    'tune_results': tune_results,
    'history': history,
}
with open(OUT / 'full_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)

with open(OUT / 'metrics.json', 'w') as f:
    json.dump({k: v for k, v in report.items() if k not in ('per_class_test', 'tune_results', 'history')},
              f, indent=2, default=str)

logger.info('=' * 70)
logger.info('FINAL SUMMARY')
logger.info('=' * 70)
logger.info(f'Backbone            : {FINAL["backbone"]} @ {IMG}px')
logger.info(f'Eff. batch size     : {EFF_BATCH} (BATCH={BATCH} × ACCUM={ACCUM})')
logger.info(f'Optimizer / LR      : AdamW @ initial {FINAL["lr"]:.1e} (CosineWarmRestarts)')
logger.info(f'Regularization      : WD={FINAL["weight_decay"]:.1e}, Dropout={FINAL["dropout"]}, LabelSmooth={FINAL["label_smoothing"]}, GradClip=1.0')
logger.info(f'Epochs              : {len(history)} / {FINAL["epochs"]} planned (best ep={best_epoch})')
logger.info(f'Early stopping      : patience={FINAL["patience"]}; triggered? {"YES" if stale >= FINAL["patience"] else "NO"}')
logger.info(f'Train samples       : {len(train_lines)}')
logger.info(f'---')
logger.info(f'Train Acc (eval)    : {train_acc_f:.4f}')
logger.info(f'Val   Acc (best)    : {best_val_acc:.4f}')
logger.info(f'Test  Acc (unseen)  : {test_acc:.4f}')
logger.info(f'Test  Macro F1      : {macro_f1:.4f}')
logger.info(f'Test  Weighted F1   : {weighted_f1:.4f}')
logger.info(f'---')
logger.info(f'Overfitting (Train→Test gap): {overfit_gap_tt:.2f} pp  →  {level}')
logger.info(f'Artifacts:')
logger.info(f'  Model     : {final_model_path}')
logger.info(f'  Report    : {OUT / "full_report.json"}')
logger.info(f'  Graph (1) : {OUT / "graphs" / "training_curves.png"}')
logger.info(f'  Graph (2) : {OUT / "graphs" / "confusion_matrix.png"}')
logger.info(f'  Config    : {OUT / "final_config.json"}')
logger.info(f'  Log       : {OUT / "logs" / "full_training.log"}')
logger.info('=' * 70)
logger.info('TRAINING COMPLETE')
