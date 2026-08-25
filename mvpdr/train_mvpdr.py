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

ROOT = Path(r"c:/Users/PANKAJ YADAV/OneDrive/Desktop/Terramind AI/data/plantwild/plantwild")
OUT = Path(r"c:/Users/PANKAJ YADAV/OneDrive/Desktop/Terramind AI/outputs")
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

# create stratified train/val split (80/20)
seed = 42
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
    'backbone':'resnet50',
    'lr': 0.001,
    'augment_epoch':1,
    'train_epoch':5,
    'batch_size':32,
    'num_workers':2,
    'seed':seed
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

# initial smoke forward to validate loader
smoke_lines = train_lines[:256]
smoke_ds = PlantWildDataset(ROOT, smoke_lines, classes_map)
smoke_dl = DataLoader(smoke_ds, batch_size=8, shuffle=True, num_workers=0)
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
batch = next(iter(smoke_dl))
imgs, labels = batch
imgs = imgs.to(device)
labels = torch.as_tensor(labels).to(device)
with torch.no_grad():
    out = model(imgs)
logger.info('smoke forward ok, output shape %s', str(out.shape))

# LR trials on tiny subset
logger.info('Beginning LR trials...')
trial_lines = train_lines[:max(200, int(0.05*len(train_lines)))]
trial_ds = PlantWildDataset(ROOT, trial_lines, classes_map)
trial_dl = DataLoader(trial_ds, batch_size=min(8, cfg['batch_size']), shuffle=True, num_workers=0)
best_lr=None; best_acc=-1.0
for lr in [0.001, 0.0005, 0.0001]:
    tmodel = models.resnet50(pretrained=True)
    tmodel.fc = nn.Linear(tmodel.fc.in_features, num_classes)
    tmodel = tmodel.to(device)
    opt = optim.Adam(tmodel.parameters(), lr=lr)
    loss, acc = train_epoch(tmodel, trial_dl, opt, device)
    logger.info('LR %s -> train acc %.4f loss %.4f', lr, acc, loss)
    if acc>best_acc:
        best_acc=acc; best_lr=lr

if best_lr is None:
    best_lr = cfg['lr']
logger.info('Selected LR: %s', best_lr)
cfg['lr']=best_lr

# Choose final training subset based on device
if device.type=='cpu':
    frac = 0.3
    n = int(len(train_lines)*frac)
    final_train_lines = train_lines[:n]
else:
    final_train_lines = train_lines
final_val_lines = val_lines

train_ds = PlantWildDataset(ROOT, final_train_lines, classes_map)
val_ds = PlantWildDataset(ROOT, final_val_lines, classes_map)
train_dl = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=min(cfg['num_workers'],4))
val_dl = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=0)

# full model and optimizer
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
opt = optim.Adam(model.parameters(), lr=cfg['lr'])

best_val=-1.0
metrics = {'history':[]}
for epoch in range(1, cfg['train_epoch']+1):
    t0=time.time()
    train_loss, train_acc = train_epoch(model, train_dl, opt, device)
    val_loss, val_acc = eval_model(model, val_dl, device)
    elapsed = time.time()-t0
    logger.info('Epoch %s/%s: train_acc=%.4f val_acc=%.4f time=%.1fs', epoch, cfg['train_epoch'], train_acc, val_acc, elapsed)
    metrics['history'].append({'epoch':epoch,'train_loss':train_loss,'train_acc':train_acc,'val_loss':val_loss,'val_acc':val_acc,'time':elapsed})
    # save checkpoint
    ckpt_path = OUT/'checkpoints'/f'epoch_{epoch}.pth'
    torch.save({'epoch':epoch,'model_state':model.state_dict(),'opt_state':opt.state_dict()}, ckpt_path)
    if val_acc>best_val:
        best_val=val_acc
        best_path = OUT/'checkpoints'/'best_checkpoint.pth'
        torch.save({'epoch':epoch,'model_state':model.state_dict(),'opt_state':opt.state_dict(),'val_acc':val_acc}, best_path)

    # update globals so save_on_exit can persist state
    current_model = model
    current_opt = opt

# final save
final_path = Path('c:/Users/PANKAJ YADAV/OneDrive/Desktop/Terramind AI/models')
final_path.mkdir(parents=True, exist_ok=True)
try:
    torch.save(model.state_dict(), final_path/'final_model.pth')
    with open(OUT/'metrics.json','w') as f:
        json.dump(metrics, f)
    logger.info('Wrote final metrics.json')
    with open(OUT/'logs'/'training.log','a') as f:
        f.write('training complete\n')
    logger.info('Training complete. best val %s', best_val)
except Exception:
    logger.exception('Error finalizing training')
