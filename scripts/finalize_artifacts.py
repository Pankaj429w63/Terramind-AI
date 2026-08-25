from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
DEST = OUT / 'final_artifacts.zip'

def collect_files(root:Path):
    files = []
    for p in root.rglob('*'):
        if p.is_file():
            files.append(p)
    return files

files = collect_files(OUT)
with zipfile.ZipFile(DEST, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, f.relative_to(ROOT))

print('Created', DEST)

if __name__ == '__main__':
    pass
