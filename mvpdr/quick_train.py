from pathlib import Path
import random
import time
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as T
import yaml
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets.plantwild import PlantWildDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / 'data' / 'plantwild' / 'plantwild'
OUT = PROJECT_ROOT / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)
(OUT/"logs").mkdir(exist_ok=True)
(OUT/"checkpoints").mkdir(exist_ok=True)

# load classes
classes_map = {}
with open(ROOT/'classes.txt','r',encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        idx,name = line.strip().split(' ',1)
        classes_map[name]=int(idx)
num_classes = len(classes_map)

with open(ROOT/'trainval.txt','r',encoding='utf-8') as f:
    all_lines=[l.strip() for l in f if l.strip()]

parser = argparse.ArgumentParser(description='Fast, regularized PlantWild training run.')
parser.add_argument('--max-train-samples', type=int, default=256)
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--patience', type=int, default=1)
args = parser.parse_args()

# stratified split
seed=42; random.seed(seed)
by_class={}
for ln in all_lines:
    cls=ln.split('=')[0].split('/')[0]
    by_class.setdefault(cls,[]).append(ln)
train_lines=[]; val_lines=[]
for cls,items in by_class.items():
    random.shuffle(items)
    cut=int(len(items)*0.8)
    train_lines+=items[:cut]
    val_lines+=items[cut:]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device',device)

cfg={'lr':0.0003,'weight_decay':0.0001,'train_epoch':args.epochs,
     'batch_size':8,'seed':seed,'patience':args.patience}
# reduce for CPU
if device.type=='cpu':
    cfg['batch_size']=8

def stratified_limit(lines, limit):
    grouped = {}
    for line in lines:
        grouped.setdefault(line.split('=')[0].split('/')[0], []).append(line)
    selected = [items[0] for items in grouped.values()]
    pool = [line for items in grouped.values() for line in items[1:]]
    random.Random(seed).shuffle(pool)
    selected.extend(pool[:max(0, limit - len(selected))])
    random.Random(seed).shuffle(selected)
    return selected

train_subset=stratified_limit(train_lines, min(args.max_train_samples, len(train_lines)))
val_subset=val_lines

train_transform = T.Compose([
    T.Resize((224,224)), T.RandomHorizontalFlip(), T.RandomRotation(10),
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    T.ToTensor(), T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])
val_transform = T.Compose([
    T.Resize((224,224)), T.ToTensor(),
    T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])
train_ds=PlantWildDataset(ROOT, train_subset, classes_map, transform=train_transform)
val_ds=PlantWildDataset(ROOT, val_subset, classes_map, transform=val_transform)
train_dl=DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=0)
val_dl=DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=0)

model=models.resnet18(weights=None)
model.fc=nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, num_classes))
model=model.to(device)
opt=optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
scheduler=optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=1)

crit=nn.CrossEntropyLoss()
metrics={'history':[]}
best_val=-1.0
stale_epochs=0
for epoch in range(1,cfg['train_epoch']+1):
    t0=time.time(); model.train(); total=0;correct=0;loss_sum=0.0
    for imgs,labels in train_dl:
        imgs=imgs.to(device); labels=torch.as_tensor(labels).to(device)
        out=model(imgs); loss=crit(out,labels)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum+=loss.item()*imgs.size(0)
        preds=out.argmax(1); correct+=(preds==labels).sum().item(); total+=imgs.size(0)
    train_loss=train_loss=loss_sum/total; train_acc=correct/total
    # val
    model.eval(); total=0;correct=0;loss_sum=0.0
    with torch.no_grad():
        for imgs,labels in val_dl:
            imgs=imgs.to(device); labels=torch.as_tensor(labels).to(device)
            out=model(imgs); loss=crit(out,labels)
            loss_sum+=loss.item()*imgs.size(0)
            preds=out.argmax(1); correct+=(preds==labels).sum().item(); total+=imgs.size(0)
    val_loss=loss_sum/total; val_acc=correct/total; elapsed=time.time()-t0
    print(f'Epoch {epoch}/{cfg["train_epoch"]}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} time={elapsed:.1f}s')
    metrics['history'].append({'epoch':epoch,'train_loss':train_loss,'train_acc':train_acc,'val_loss':val_loss,'val_acc':val_acc,'time':elapsed})
    scheduler.step(val_loss)
    ckpt=OUT/'checkpoints'/f'quick_epoch_{epoch}.pth'
    torch.save({'epoch':epoch,'state':model.state_dict(),'opt':opt.state_dict(),'val_acc':val_acc}, ckpt)
    if val_acc>best_val:
        best_val=val_acc; torch.save({'state':model.state_dict(),'val_acc':val_acc}, OUT/'checkpoints'/'quick_best.pth')
        stale_epochs=0
    else:
        stale_epochs += 1
        if stale_epochs >= cfg['patience']:
            print(f'Early stopping at epoch {epoch}')
            break

# save final
models_dir = PROJECT_ROOT / 'models'
models_dir.mkdir(parents=True, exist_ok=True)
best_checkpoint = torch.load(OUT/'checkpoints'/'quick_best.pth', map_location=device)
model.load_state_dict(best_checkpoint['state'])
torch.save({'model_state':model.state_dict(), 'num_classes':num_classes,
            'classes':classes_map, 'best_val_acc':best_val},
           models_dir/'quick_final.pth')
with open(OUT/'metrics.json','w') as f: json.dump(metrics,f)
with open(OUT/'logs'/'training.log','a') as f: f.write('quick run complete\n')
print('Quick run complete. best val', best_val)
