from pathlib import Path
import random, json, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets.plantwild import PlantWildDataset

ROOT=Path(r"c:/Users/PANKAJ YADAV/OneDrive/Desktop/Terramind AI/data/plantwild/plantwild")
OUT=Path(r"c:/Users/PANKAJ YADAV/OneDrive/Desktop/Terramind AI/outputs")
OUT.mkdir(parents=True,exist_ok=True); (OUT/"logs").mkdir(exist_ok=True); (OUT/"checkpoints").mkdir(exist_ok=True)

with open(ROOT/'classes.txt','r',encoding='utf-8') as f:
    classes=[l.strip() for l in f if l.strip()]

with open(ROOT/'trainval.txt','r',encoding='utf-8') as f:
    all_lines=[l.strip() for l in f if l.strip()]

# tiny subset
train_lines=all_lines[:100]
val_lines=all_lines[100:140]

device=torch.device('cpu')
print('Device',device)

train_ds=PlantWildDataset(ROOT, train_lines, {c.split(' ',1)[1]:int(c.split(' ',1)[0]) for c in classes})
val_ds=PlantWildDataset(ROOT, val_lines, {c.split(' ',1)[1]:int(c.split(' ',1)[0]) for c in classes})
train_dl=DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
val_dl=DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

# tiny CNN to avoid heavy native BLAS/MKL operations
class TinyCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(16, num_classes)
    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = TinyCNN(len(classes)).to(device)
opt = optim.Adam(model.parameters(), lr=0.001)
crit = nn.CrossEntropyLoss()

metrics={'history':[]}
try:
    for epoch in range(1,2):
        t0=time.time(); model.train(); total=0;correct=0;loss_sum=0.0
        for imgs,labels in train_dl:
            imgs=imgs.to(device); labels=torch.as_tensor(labels).to(device)
            out=model(imgs); loss=crit(out,labels); opt.zero_grad(); loss.backward(); opt.step()
            loss_sum+=loss.item()*imgs.size(0); preds=out.argmax(1); correct+=(preds==labels).sum().item(); total+=imgs.size(0)
        train_loss=loss_sum/total if total>0 else 0.0; train_acc=correct/total if total>0 else 0.0
        # val
        model.eval(); total=0;correct=0;loss_sum=0.0
        with torch.no_grad():
            for imgs,labels in val_dl:
                imgs=imgs.to(device); labels=torch.as_tensor(labels).to(device)
                out=model(imgs); loss=crit(out,labels); loss_sum+=loss.item()*imgs.size(0); preds=out.argmax(1); correct+=(preds==labels).sum().item(); total+=imgs.size(0)
        val_loss=loss_sum/total if total>0 else 0.0; val_acc=correct/total if total>0 else 0.0; elapsed=time.time()-t0
        print(f'Epoch {epoch}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} time={elapsed:.1f}s')
        metrics['history'].append({'epoch':epoch,'train_loss':train_loss,'train_acc':train_acc,'val_loss':val_loss,'val_acc':val_acc,'time':elapsed})
        ckpt=OUT/'checkpoints'/f'tiny_epoch_{epoch}.pth'
        torch.save({'epoch':epoch,'state':model.state_dict(),'opt':opt.state_dict(),'val_acc':val_acc}, ckpt)
        # save lightweight final model
        final_dir = Path('c:/Users/PANKAJ YADAV/OneDrive/Desktop/Terramind AI/models')
        final_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), final_dir/'tiny_final.pth')
    with open(OUT/'metrics.json','w') as f: json.dump(metrics,f)
    with open(OUT/'logs'/'training.log','a') as f: f.write('tiny run complete\n')
    print('Tiny run complete')
except Exception as e:
    # ensure we write logs and metrics even on failure
    try:
        with open(OUT/'metrics.json','w') as f: json.dump(metrics,f)
    except Exception:
        pass
    print('Tiny run failed:', e)
