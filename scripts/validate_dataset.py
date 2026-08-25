from pathlib import Path
import json
from PIL import Image
import hashlib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'metrics'
OUT.mkdir(parents=True, exist_ok=True)

def discover_datasets(data_root: Path):
    datasets = []
    # look for common dataset folders
    for p in data_root.rglob('*'):
        if p.is_dir() and (p / 'trainval.txt').exists():
            datasets.append(p)
    # also check common direct path
    if (data_root / 'plantwild' / 'plantwild').exists():
        datasets.append((data_root / 'plantwild' / 'plantwild'))
    # de-dup
    uniq = []
    for d in datasets:
        if d not in uniq:
            uniq.append(d)
    return uniq

def hash_file(path: Path, block_size=65536):
    h = hashlib.sha1()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            h.update(block)
    return h.hexdigest()

def analyze_dataset(ds_root: Path):
    report = {'dataset_path': str(ds_root), 'found_images': 0, 'classes': {}, 'missing_paths': [], 'corrupt_files': [], 'sample_opened': []}
    images_dir = ds_root / 'images'
    # parse classes.txt if present
    classes_map = {}
    if (ds_root / 'classes.txt').exists():
        for ln in (ds_root / 'classes.txt').read_text(encoding='utf-8').splitlines():
            if not ln.strip():
                continue
            idx, name = ln.strip().split(' ',1)
            classes_map[name] = int(idx)
    # collect image paths from trainval.txt if present
    lines = []
    if (ds_root / 'trainval.txt').exists():
        lines = [l.strip() for l in (ds_root / 'trainval.txt').read_text(encoding='utf-8').splitlines() if l.strip()]
    # iterate
    hashes = {}
    for ln in lines:
        parts = ln.split('=')
        rel = parts[0]
        # ensure path under images/
        if not Path(rel).parts[0].lower().startswith('images'):
            rel = 'images/' + rel
        p = ds_root / rel
        if not p.exists():
            report['missing_paths'].append(str(p))
            continue
        report['found_images'] += 1
        # determine class
        cls = Path(parts[0]).parts[0]
        report['classes'].setdefault(cls, 0)
        report['classes'][cls] += 1
        # try opening (sample up to 3 per class)
        try:
            with Image.open(p) as im:
                im.verify()
            # store sample path
            if len(report['sample_opened']) < 20:
                report['sample_opened'].append(str(p))
            # compute hash for duplicate detection (limited)
            h = hash_file(p)
            hashes.setdefault(h, []).append(str(p))
        except Exception as e:
            report['corrupt_files'].append({'path': str(p), 'error': str(e)})

    # detect duplicates
    dup = {h:ps for h,ps in hashes.items() if len(ps)>1}
    report['duplicates'] = dup
    return report

def main():
    data_root = ROOT / 'data'
    datasets = discover_datasets(data_root)
    full_report = {'datasets': []}
    if not datasets:
        # fallback: check data/plantwild/plantwild
        p = data_root / 'plantwild' / 'plantwild'
        if p.exists():
            datasets = [p]
    for ds in datasets:
        r = analyze_dataset(ds)
        full_report['datasets'].append(r)
    # write report
    outp = OUT / 'dataset_report.json'
    with outp.open('w', encoding='utf-8') as f:
        json.dump(full_report, f, indent=2)
    print('Wrote', outp)

if __name__ == '__main__':
    main()
