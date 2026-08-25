from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
models = ROOT / 'models'

def check():
    ok = True
    artifacts = [OUT / 'configs_effective.yaml', OUT / 'logs' / 'training.log', OUT / 'checkpoints', OUT / 'metrics.json', models / 'final_model.pth']
    for a in artifacts:
        if not a.exists():
            print('MISSING:', a)
            ok = False
        else:
            print('FOUND :', a)
    if ok:
        with open(OUT / 'metrics.json') as f:
            data = json.load(f)
        print('\nMetrics summary keys:', list(data.keys())[:10])
    return ok

if __name__ == '__main__':
    import sys
    ok = check()
    sys.exit(0 if ok else 2)
