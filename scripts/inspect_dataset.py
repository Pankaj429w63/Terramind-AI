import os
from pathlib import Path
root = Path(r"c:/Users/PANKAJ YADAV/OneDrive/Desktop/Terramind AI/data/plantwild/plantwild")
print('dataset_root:', root)
classes_file = root/ 'classes.txt'
print('classes_file_exists:', classes_file.exists())
with open(classes_file,'r',encoding='utf-8') as f:
    classes = [l.strip() for l in f if l.strip()]
print('num_classes:', len(classes))
# count images
images_dir = root / 'images'
num_images = 0
for p,dirs,files in os.walk(images_dir):
    for fn in files:
        if fn.lower().endswith(('.jpg','.jpeg','.png','.bmp')):
            num_images += 1
print('num_images:', num_images)
# check splits
trainval = root / 'trainval.txt'
print('trainval_exists:', trainval.exists())
if trainval.exists():
    with open(trainval,'r',encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    print('trainval_lines:', len(lines))
    print('trainval_sample_3:', lines[:3])
    print('trainval_sample_10:', lines[:10])
else:
    print('trainval missing')
# test.txt
test = root / 'test.txt'
print('test_exists:', test.exists())
# list 10 image paths from samples
print('\nSample image checks (first 10 entries in trainval):')
for i,ln in enumerate(lines[:10]):
    # format like "apple black rot/google_0082.jpg=0=1"
    parts = ln.split('=')
    imgpath = parts[0]
    p = root / imgpath
    exists = p.exists()
    print(i+1, imgpath, '->', exists)
