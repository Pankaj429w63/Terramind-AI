from pathlib import Path
import json, time
import torch
import torch.nn as nn
import torch.optim as optim

OUT=Path(r"c:/Users/PANKAJ YADAV/OneDrive/Desktop/Terramind AI/outputs")
OUT.mkdir(parents=True, exist_ok=True)
(OUT/"logs").mkdir(exist_ok=True)
(OUT/"checkpoints").mkdir(exist_ok=True)

device = torch.device('cpu')
print('Device', device)

# tiny synthetic model
class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
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

model = TinyCNN(num_classes=10).to(device)
opt = optim.SGD(model.parameters(), lr=0.01)
crit = nn.CrossEntropyLoss()

# synthetic data: 200 samples
N=200; B=16
metrics={'history':[]}
start=time.time()
for epoch in range(1,2):
    t0=time.time(); model.train(); total=0; correct=0; loss_sum=0.0
    for i in range(0, N, B):
        bs = min(B, N - i)
        imgs = torch.randn(bs,3,64,64).to(device)
        labels = torch.randint(0,10,(bs,)).to(device)
        out = model(imgs)
        loss = crit(out, labels)
        opt.zero_grad(); loss.backward(); opt.step()
        preds = out.argmax(1)
        correct += (preds==labels).sum().item(); total += bs; loss_sum += loss.item()*bs
    train_loss = loss_sum/total; train_acc = correct/total
    elapsed = time.time()-t0
    print(f'Epoch {epoch}: train_acc={train_acc:.4f} train_loss={train_loss:.4f} time={elapsed:.2f}s')
    metrics['history'].append({'epoch':epoch,'train_loss':train_loss,'train_acc':train_acc,'time':elapsed})
    ck = OUT/'checkpoints'/f'synth_epoch_{epoch}.pth'
    torch.save({'epoch':epoch,'model_state':model.state_dict(),'opt_state':opt.state_dict()}, ck)

# final save
models_dir = Path(r"c:/Users/PANKAJ YADAV/OneDrive/Desktop/Terramind AI/models")
models_dir.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), models_dir/'synth_final.pth')
with open(OUT/'metrics.json','w') as f: json.dump(metrics,f)
with open(OUT/'logs'/'training.log','a') as f: f.write('synthetic run complete\n')
print('Synthetic run complete in', time.time()-start)
