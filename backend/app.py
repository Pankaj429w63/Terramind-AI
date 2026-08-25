from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import io
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title='Terramind-AI Inference')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


def load_labels():
    # try to find classes.txt in data
    p = ROOT / 'data' / 'plantwild' / 'plantwild' / 'classes.txt'
    labels = None
    if p.exists():
        labels = []
        for ln in p.read_text(encoding='utf-8').splitlines():
            if not ln.strip():
                continue
            parts = ln.strip().split(' ', 1)
            if len(parts) == 2:
                labels.append(parts[1])
            else:
                labels.append(parts[0])
    return labels


class TinyFallback(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.fc = torch.nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def try_load_model(device='cpu'):
    # Prefer ConvNeXt production, then EfficientNet baseline, then final_model, then tiny
    candidates = [MODEL_DIR / 'convnext_tiny_final.pth', MODEL_DIR / 'efficientnet_b0_final.pth', MODEL_DIR / 'final_model.pth', MODEL_DIR / 'tiny_final.pth']
    for c in candidates:
        if c.exists():
            try:
                ck = torch.load(c, map_location=device)
                # checkpoint may be dict with keys: 'state', 'state_dict', 'model', or raw state_dict
                state = None
                if isinstance(ck, dict):
                    if 'model' in ck and isinstance(ck['model'], torch.nn.Module):
                        model = ck['model']
                        model.to(device)
                        model.eval()
                        return model, c.name
                    for k in ('state','state_dict','model_state'):
                        if k in ck:
                            state = ck[k]
                            break
                    if state is None:
                        # maybe the dict itself is a state_dict
                        state = ck
                else:
                    state = ck
                labels = load_labels() or []
                model = TinyFallback(num_classes=max(2, len(labels) or 10))
                model.load_state_dict(state)
                model.to(device)
                model.eval()
                return model, c.name
            except Exception:
                continue
    # fallback: fresh tiny model
    labels = load_labels() or []
    model = TinyFallback(num_classes=max(2, len(labels) or 10))
    model.eval()
    return model, 'tiny_fallback'


DEVICE = 'cpu'
MODEL, MODEL_NAME = try_load_model(DEVICE)
LABELS = load_labels()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@app.get('/health')
def health():
    return {'status': 'ok', 'model': MODEL_NAME}


@app.get('/labels')
def labels():
    if LABELS:
        return {'labels': LABELS}
    return {'labels': []}


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('jpg', 'jpeg', 'png')):
        raise HTTPException(status_code=400, detail='Unsupported file type')
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Invalid image: {e}')
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = MODEL(x)
        probs = F.softmax(out, dim=1).squeeze(0).cpu().tolist()
    # map top-k
    topk = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:5]
    preds = []
    for idx, p in topk:
        lab = LABELS[idx] if LABELS and idx < len(LABELS) else str(idx)
        preds.append({'label': lab, 'score': float(p)})
    return JSONResponse({'predictions': preds, 'model': MODEL_NAME})
