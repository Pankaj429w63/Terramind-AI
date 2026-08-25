from pathlib import Path
import time, json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets.plantwild import PlantWildDataset
from torchvision import transforms
import torchvision.models as models

ROOT=Path(r"c:/Users/PANKAJ YADAV/OneDrive/Desktop/Terramind AI/data/plantwild/plantwild")
OUT=Path(r"c:/Users/PANKAJ YADAV/OneDrive/Desktop/Terramind AI/outputs")
OUT.mkdir(parents=True,exist_ok=True); (OUT/"logs").mkdir(exist_ok=True); (OUT/"checkpoints").mkdir(exist_ok=True)

with open(ROOT/'classes.txt','r',encoding='utf-8') as f:
    classes=[l.strip() for l in f if l.strip()]
with open(ROOT/'trainval.txt','r',encoding='utf-8') as f:
    all_lines=[l.strip() for l in f if l.strip()]

# small subset for quick CPU-safe test
train_lines=all_lines[:200]
val_lines=all_lines[200:260]

device=torch.device('cpu')

transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

train_ds=PlantWildDataset(ROOT, train_lines, {c.split(' ',1)[1]:int(c.split(' ',1)[0]) for c in classes}, transform=transform)
val_ds=PlantWildDataset(ROOT, val_lines, {c.split(' ',1)[1]:int(c.split(' ',1)[0]) for c in classes}, transform=transform)
train_dl=DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
val_dl=DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

def build_model(name, num_classes):
    if name=='efficientnet_b0':
        m = models.efficientnet_b0(pretrained=False)
        in_feats = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_feats, num_classes)
        return m
    if name=='convnext_tiny':
        m = models.convnext_tiny(pretrained=False)
        in_feats = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(in_feats, num_classes)
        return m
    raise ValueError('unknown model')

def train_one(model_name):
    model = build_model(model_name, len(classes)).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
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
            print(f'{model_name} Epoch {epoch}: train_acc={train_acc:.4f} val_acc={val_acc:.4f} time={elapsed:.1f}s')
            metrics['history'].append({'epoch':epoch,'train_loss':train_loss,'train_acc':train_acc,'val_loss':val_loss,'val_acc':val_acc,'time':elapsed})
            ckpt=OUT/'checkpoints'/f'{model_name}_epoch_{epoch}.pth'
            torch.save({'epoch':epoch,'state':model.state_dict(),'opt':opt.state_dict(),'val_acc':val_acc}, ckpt)
            final_dir = Path('c:/Users/PANKAJ YADAV/OneDrive/Desktop/Terramind AI/models')
            final_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), final_dir/f'{model_name}_final.pth')
        with open(OUT/f'metrics_{model_name}.json','w') as f: json.dump(metrics,f)
        with open(OUT/'logs'/'training.log','a') as f: f.write(f'{model_name} run complete\n')
        print(f'{model_name} run complete')
    except Exception as e:
        try:
            with open(OUT/f'metrics_{model_name}.json','w') as f: json.dump(metrics,f)
        except Exception:
            pass
        print(f'{model_name} run failed:', e)

if __name__=='__main__':
    # run baseline then production model
    train_one('efficientnet_b0')
    train_one('convnext_tiny')
