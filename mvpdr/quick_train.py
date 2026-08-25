from pathlib import Path
import random
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import yaml
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets.plantwild import PlantWildDataset

ROOT = Path(r"c:/Users/PANKAJ YADAV/OneDrive/Desktop/Terramind AI/data/plantwild/plantwild")
OUT = Path(r"c:/Users/PANKAJ YADAV/OneDrive/Desktop/Terramind AI/outputs")
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

# stratified
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

cfg={'lr':0.001,'train_epoch':2,'batch_size':8,'seed':seed}
# reduce for CPU
if device.type=='cpu':
    cfg['batch_size']=8

# use tiny subset 10% for quick run
n=int(len(train_lines)*0.1)
train_subset=train_lines[:n]
val_subset=val_lines[:min(200,len(val_lines))]

train_ds=PlantWildDataset(ROOT, train_subset, classes_map)
val_ds=PlantWildDataset(ROOT, val_subset, classes_map)
train_dl=DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=0)
val_dl=DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=0)

model=models.resnet50(pretrained=True)
model.fc=nn.Linear(model.fc.in_features, num_classes)
model=model.to(device)
opt=optim.Adam(model.parameters(), lr=cfg['lr'])

crit=nn.CrossEntropyLoss()
metrics={'history':[]}
best_val=-1.0
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
    ckpt=OUT/'checkpoints'/f'quick_epoch_{epoch}.pth'
    torch.save({'epoch':epoch,'state':model.state_dict(),'opt':opt.state_dict(),'val_acc':val_acc}, ckpt)
    if val_acc>best_val:
        best_val=val_acc; torch.save({'state':model.state_dict(),'val_acc':val_acc}, OUT/'checkpoints'/'quick_best.pth')

# save final
Path('c:/Users/PANKAJ YADAV/OneDrive/Desktop/Terramind AI/models').mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), Path('c:/Users/PANKAJ YADAV/OneDrive/Desktop/Terramind AI/models')/'quick_final.pth')
with open(OUT/'metrics.json','w') as f: json.dump(metrics,f)
with open(OUT/'logs'/'training.log','a') as f: f.write('quick run complete\n')
print('Quick run complete. best val', best_val)
