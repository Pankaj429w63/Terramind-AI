import os
import json
from pathlib import Path
import random
import time
import yaml
import sys
import atexit
import signal
import logging
import traceback
import argparse

# limit native threaded libraries to 1 thread to avoid Fortran/MKL aborts
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('NUMEXPR_NUM_THREADS','1')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as T

# limit PyTorch threads
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

# create a module-level logger early so any early checks can log safely
logger = logging.getLogger('mvpdr')
if not logger.handlers:
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(ch)

# ensure local package import
sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets.plantwild import PlantWildDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / 'data' / 'plantwild' / 'plantwild'
OUT = PROJECT_ROOT / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)
(OUT/"logs").mkdir(exist_ok=True)
(OUT/"checkpoints").mkdir(exist_ok=True)

# configure logging to file (keep existing console handler)
log_path = OUT/'logs'/'training.log'
try:
    fh = logging.FileHandler(log_path, mode='a')
    fh.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logging.getLogger().addHandler(fh)
except Exception:
    # If file handler cannot be created, keep console handler and continue
    logger.exception('Unable to create file handler for logging; continuing with console only')

# load classes
classes_file = ROOT / 'classes.txt'
classes_map = {}
with open(classes_file,'r',encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        idx,name = line.strip().split(' ',1)
        classes_map[name] = int(idx)
num_classes = len(classes_map)

# load trainval lines
with open(ROOT/'trainval.txt','r',encoding='utf-8') as f:
    all_lines = [l.strip() for l in f if l.strip()]

parser = argparse.ArgumentParser(description='Train and tune the PlantWild classifier.')
parser.add_argument('--max-train-samples', type=int, default=1200)
parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--patience', type=int, default=2)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

# create a reproducible stratified train/val split (80/20)
seed = args.seed
random.seed(seed)
by_class = {}
for ln in all_lines:
    cls = ln.split('=')[0].split('/')[0]
    by_class.setdefault(cls,[]).append(ln)
train_lines=[]
val_lines=[]
for cls,items in by_class.items():
    random.shuffle(items)
    cut = int(len(items)*0.8)
    train_lines += items[:cut]
    val_lines += items[cut:]

# quick device detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info('Device: %s', device)

# hyperparams
cfg = {
    'backbone':'resnet18',
    'lr': 0.0003,
    'weight_decay': 0.0001,
    'augment_epoch':1,
    'train_epoch':args.epochs,
    'batch_size':32,
    'num_workers':2,
    'seed':seed,
    'patience':args.patience,
    'max_train_samples':args.max_train_samples,
}
# adapt batch size for CPU
if device.type=='cpu':
    cfg['batch_size'] = 16

# define helpers

def train_epoch(model, dl, opt, device):
    model.train()
    total=0; correct=0; loss_sum=0.0
    crit = nn.CrossEntropyLoss()
    for imgs, labels in dl:
        imgs = imgs.to(device)
        labels = torch.as_tensor(labels).to(device)
        out = model(imgs)
        loss = crit(out, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_sum += loss.item()*imgs.size(0)
        preds = out.argmax(1)
        correct += (preds==labels).sum().item()
        total += imgs.size(0)
    return loss_sum/total, correct/total


def eval_model(model, dl, device):
    model.eval()
    total=0; correct=0; loss_sum=0.0
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for imgs, labels in dl:
            imgs = imgs.to(device)
            labels = torch.as_tensor(labels).to(device)
            out = model(imgs)
            loss = crit(out, labels)
            loss_sum += loss.item()*imgs.size(0)
            preds = out.argmax(1)
            correct += (preds==labels).sum().item()
            total += imgs.size(0)
    return loss_sum/total, correct/total

# save effective config placeholder (will be updated later)
with open(OUT/'configs_effective.yaml','w') as f:
    yaml.safe_dump({'cfg':cfg,'device':str(device)}, f)

# graceful shutdown helpers
current_model = None
current_opt = None
metrics = {'history':[]}
best_val = -1.0

def save_on_exit(reason=None):
    try:
        logger.info('Saving state on exit. Reason: %s', str(reason))
        if current_model is not None:
            ck = OUT/'checkpoints'/'interrupted_checkpoint.pth'
            try:
                opt_state = current_opt.state_dict() if current_opt is not None else None
            except Exception:
                opt_state = None
            torch.save({'model_state': current_model.state_dict(), 'opt_state': opt_state}, ck)
            logger.info('Saved interrupted checkpoint to %s', str(ck))
        try:
            with open(OUT/'metrics.json','w') as mf:
                json.dump(metrics, mf)
            logger.info('Wrote metrics.json')
        except Exception:
            logger.exception('Failed writing metrics.json')
    except Exception:
        logger.exception('Error during save_on_exit')

def _signal_handler(sig, frame):
    save_on_exit(f'signal {sig}')
    raise SystemExit()

atexit.register(save_on_exit)
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# Use a single model factory so tuning and final training are comparable.
def make_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model.to(device)

def stratified_limit(lines, limit):
    if limit >= len(lines):
        return list(lines)
    grouped = {}
    for line in lines:
        cls = line.split('=')[0].split('/')[0]
        grouped.setdefault(cls, []).append(line)
    selected = []
    # Keep every class represented, then distribute remaining capacity by class size.
    for items in grouped.values():
        selected.append(items[0])
    remaining = max(0, limit - len(selected))
    pool = [item for items in grouped.values() for item in items[1:]]
    random.Random(seed).shuffle(pool)
    selected.extend(pool[:remaining])
    random.Random(seed).shuffle(selected)
    return selected

# LR trials must be scored on validation data, not training accuracy.
logger.info('Beginning LR trials...')
trial_lines = stratified_limit(train_lines, min(args.max_train_samples, len(train_lines)))
trial_ds = PlantWildDataset(ROOT, trial_lines, classes_map)
trial_dl = DataLoader(trial_ds, batch_size=min(8, cfg['batch_size']), shuffle=True, num_workers=0)
val_ds = PlantWildDataset(ROOT, val_lines, classes_map)
val_dl = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=0)
best_lr=None; best_val=-1.0
trials = []
for lr in [0.0001, 0.0003, 0.001]:
    tmodel = make_model()
    opt = optim.AdamW(tmodel.parameters(), lr=lr, weight_decay=cfg['weight_decay'])
    train_loss, train_acc = train_epoch(tmodel, trial_dl, opt, device)
    val_loss, val_acc = eval_model(tmodel, val_dl, device)
    result = {'lr':lr, 'train_loss':train_loss, 'train_acc':train_acc,
              'val_loss':val_loss, 'val_acc':val_acc}
    trials.append(result)
    logger.info('LR %s -> train_acc %.4f val_acc %.4f val_loss %.4f',
                lr, train_acc, val_acc, val_loss)
    if val_acc > best_val:
        best_val=val_acc; best_lr=lr

if best_lr is None:
    best_lr = cfg['lr']
logger.info('Selected LR: %s', best_lr)
cfg['lr']=best_lr

# Keep CPU runs bounded but preserve the stratified split.
final_train_lines = stratified_limit(train_lines, min(args.max_train_samples, len(train_lines)))
final_val_lines = val_lines

train_transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_ds = PlantWildDataset(ROOT, final_train_lines, classes_map, transform=train_transform)
val_ds = PlantWildDataset(ROOT, final_val_lines, classes_map, transform=val_transform)
train_dl = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=min(cfg['num_workers'],4))
val_dl = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=0)

# full model and optimizer
model = make_model()
opt = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=1)

best_val=-1.0
best_epoch = 0
stale_epochs = 0
metrics = {'config':cfg, 'lr_trials':trials, 'history':[]}
for epoch in range(1, cfg['train_epoch']+1):
    t0=time.time()
    train_loss, train_acc = train_epoch(model, train_dl, opt, device)
    val_loss, val_acc = eval_model(model, val_dl, device)
    elapsed = time.time()-t0
    logger.info('Epoch %s/%s: train_acc=%.4f val_acc=%.4f time=%.1fs', epoch, cfg['train_epoch'], train_acc, val_acc, elapsed)
    metrics['history'].append({'epoch':epoch,'train_loss':train_loss,'train_acc':train_acc,'val_loss':val_loss,'val_acc':val_acc,'time':elapsed})
    scheduler.step(val_loss)
    # save checkpoint
    ckpt_path = OUT/'checkpoints'/f'epoch_{epoch}.pth'
    torch.save({'epoch':epoch,'model_state':model.state_dict(),'opt_state':opt.state_dict()}, ckpt_path)
    if val_acc>best_val:
        best_val=val_acc
        best_epoch = epoch
        stale_epochs = 0
        best_path = OUT/'checkpoints'/'best_checkpoint.pth'
        torch.save({'epoch':epoch,'model_state':model.state_dict(),'opt_state':opt.state_dict(),'val_acc':val_acc}, best_path)
    else:
        stale_epochs += 1
        if stale_epochs >= cfg['patience']:
            logger.info('Early stopping at epoch %s; best epoch %s', epoch, best_epoch)
            break

    # update globals so save_on_exit can persist state
    current_model = model
    current_opt = opt

# final save
final_path = PROJECT_ROOT / 'models'
final_path.mkdir(parents=True, exist_ok=True)
try:
    if best_path.exists():
        best_checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(best_checkpoint['model_state'])
    torch.save({'model_state':model.state_dict(), 'num_classes':num_classes,
                'classes':classes_map, 'best_val_acc':best_val,
                'best_epoch':best_epoch}, final_path/'final_model.pth')
    with open(OUT/'metrics.json','w') as f:
        json.dump(metrics, f)
    logger.info('Wrote final metrics.json')
    with open(OUT/'logs'/'training.log','a') as f:
        f.write('training complete\n')
    logger.info('Training complete. best val %s', best_val)
except Exception:
    logger.exception('Error finalizing training')
