import sys
from pathlib import Path
from inference.predict import predict

image_path = list(Path('data/synthetic/images').glob('*.png'))[0]

# Wrap stdout to a safe UTF-8 file
with open('debug_log.txt', 'w', encoding='utf-8') as f:
    sys.stdout = f
    predict(str(image_path), 'checkpoints/best_checkpoint.pth', 'outputs/', device='cpu')
    sys.stdout = sys.__stdout__
