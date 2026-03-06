# Instance Segmentation & Quantification of Translucent Pulp Fibrils
### Swin-Transformer + Attention-Gated U-Net + Mask2Former | Precision Papermaking

---

## 📌 Project Overview
This project implements a 4-stage deep learning pipeline for:
1. **Super-Resolution** of low-contrast pulp fiber microscopy images (Real-ESRGAN)
2. **Feature Extraction** using a Swin-Transformer backbone (global context)
3. **Instance Segmentation** using Mask2Former (handles overlapping/translucent fibrils)
4. **Quantification** via Skeletonization + Graph Neural Network (length, tortuosity, Fibrillation Index)

**Dataset:** Synthetic (FibrilSynth — procedurally generated with auto-annotations)  
**Training Hardware:** Kaggle Free Tier (P100/T4, 15–16GB VRAM)

---

## 🗂️ Project Structure
```
DL CP/
├── data/
│   └── synthetic/          ← Generated images, masks, annotations.json
├── data_pipeline/
│   ├── synthetic_gen.py    ← Fibril image generator (FibrilSynth)
│   ├── generate_dataset.py ← CLI: generate N images
│   ├── dataset.py          ← PyTorch Dataset
│   └── augmentation.py     ← Albumentations pipeline
├── models/
│   ├── esrgan_sr.py        ← Stage 1: Super-resolution
│   ├── swin_backbone.py    ← Stage 2: Swin-T backbone
│   ├── ag_pixel_decoder.py ← Attention-Gated U-Net decoder
│   └── mask2former.py      ← Stage 3: Full segmentation model
├── training/
│   ├── losses.py           ← Focal + Dice + Hungarian Matcher
│   ├── train.py            ← Main training loop (Kaggle-ready)
│   └── train_esrgan.py     ← ESRGAN fine-tuning
├── quantification/
│   ├── skeletonize.py      ← Mask → skeleton → NetworkX graph
│   └── gnn_quantifier.py   ← GraphSAGE metric predictor
├── inference/
│   ├── predict.py          ← Full pipeline inference
│   ├── report_generator.py ← CSV output
│   └── visualize.py        ← Color-coded visualization
├── evaluation/
│   └── evaluate.py         ← COCO AP + domain metrics
├── tests/                  ← Unit tests
├── checkpoints/            ← Saved model weights
├── outputs/                ← Prediction outputs, CSVs
├── kaggle_train.ipynb      ← Cloud training notebook
└── requirements.txt
```

---

## ⚙️ Setup (Local — Windows)

```bash
# 1. Create a virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install PyTorch first (CPU for local dev, GPU on Kaggle)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Install all other requirements
pip install -r requirements.txt

# 4. Install torch-geometric (needed for GNN)
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

---

## 🚀 Step-by-Step Run Guide

### Step 1 — Generate Synthetic Dataset (Run Locally)
```bash
cd "d:\VIT\Sem 6\DL CP"
python data_pipeline/generate_dataset.py --n 500 --output data/synthetic --seed 42
```
- Generates 500 synthetic images with overlapping translucent fibrils
- Auto-creates `data/synthetic/annotations.json` in COCO format
- Takes ~5-10 minutes on CPU

### Step 2 — Verify Your Data
```bash
python inference/visualize.py --mode check_synthetic --n 5
```
Opens 5 random images with their masks overlaid. ✅ If masks align with fibers, you're good.

### Step 3 — Upload to Kaggle
1. Zip the entire project folder
2. Go to [kaggle.com](https://kaggle.com) → Datasets → New Dataset → upload zip
3. Open `kaggle_train.ipynb` as a new Kaggle Notebook
4. Attach your dataset
5. Enable GPU (P100 or T4)
6. Run all cells — training takes ~10–14 hours

### Step 4 — Download Checkpoint
After training completes, download `checkpoints/best_checkpoint.pth` from Kaggle output.

### Step 5 — Run Inference Locally
```bash
python inference/predict.py \
  --image data/synthetic/images/0001.png \
  --checkpoint checkpoints/best_checkpoint.pth \
  --output outputs/
```
Produces:
- `outputs/0001_segmented.png` — color-coded mask overlay
- `outputs/0001_report.csv` — fibril metrics

---

## 📊 Output Metrics (per image)
| Column | Description |
|---|---|
| `fibril_id` | Unique instance ID |
| `length_um` | Fibril length in micrometers |
| `width_um` | Average fibril width |
| `tortuosity` | Curliness (1.0 = perfectly straight) |
| `branching_count` | Number of branch points |
| `fibrillation_index` | FI score (higher = more refined) |

---

## 🔬 Architecture Summary
```
Input (1024×1024 grayscale)
  → Real-ESRGAN x2 → 1024×1024 enhanced
  → Swin-T Backbone → 4-scale feature maps
  → AG-UNet Pixel Decoder → per-pixel embeddings
  → Mask2Former Decoder (50 queries) → N instance masks
  → Skeletonize → NetworkX graph
  → GraphSAGE → metric tensor
  → CSV Report
```

---

## 📚 Key References
- Mask2Former: Cheng et al., CVPR 2022
- Swin-T: Liu et al., ICCV 2021
- Attention U-Net: Oktay et al., MIDL 2018
- Real-ESRGAN: Wang et al., ICCV 2021
- GraphSAGE: Hamilton et al., NeurIPS 2017
