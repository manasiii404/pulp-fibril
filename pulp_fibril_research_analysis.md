# 🔬 Deep Research Analysis: Instance Segmentation & Quantification of Translucent Pulp Fibrils
### Swin-Transformer + Attention-Gated U-Net + Mask2Former | Precision Papermaking
---

> **TL;DR** — This document provides a critical, rigorous breakdown of feasibility, dataset acquisition, architecture design, training strategy, evaluation metrics, novelty claims, and publication guidance for your capstone project. Everything you need to go from idea → publishable research.

---

## 1. 🧪 Problem Framing & Research Gap Analysis

### 1.1 Why This Problem Is Hard (Technically)

| Challenge | Root Cause | Why Current Methods Fail |
|---|---|---|
| **Low Contrast** | Pulp fibrils are semi-transparent (refractive index ≈ water) | CNNs rely on texture gradients — none exist here |
| **Occlusion** | Fibers cross each other in 3D space | Semantic segmentation merges crossing fibers into one blob |
| **Scale Variability** | Main fiber ~30μm wide, fibrils ~0.5–2μm | A single CNN receptive field can't span both scales simultaneously |
| **Data Scarcity** | No large public labeled dataset for pulp fibrils | Transfer learning + synthetic data needed |
| **Global Context Loss** | Patch-based CNNs process 128×128 tiles | Long fibers (200μm+) get "broken" across patch boundaries |
| **3D in 2D projection** | Microscopy collapses depth into one plane | Depth estimation from a 2D image requires intensity gradient modeling |

### 1.2 Your Research Gaps to Exploit (Where Existing Work Fails)

1. **No published work** specifically applies Mask2Former with Swin-T backbone to pulp fiber instance segmentation
2. **No DL-based super-resolution preprocessing** (Real-ESRGAN) has been used in a pulp fibrillation analysis pipeline
3. **No GNN-based tortuosity + branching analysis** applied to fibrillated pulp surfaces
4. **No pseudo-3D depth estimation** from single 2D pulp fibril images using intensity profiles
5. Papers on fiber segmentation use outdated backbones (ResNet-50, VGG) — Swin-L/B hasn't been tried

---

## 2. ✅ Feasibility Assessment

### 2.1 Technical Feasibility: STRONG ✅

| Component | Feasibility | Reason |
|---|---|---|
| Real-ESRGAN preprocessing | **Very High** | Pre-trained weights available; fine-tunable on grayscale microscopy |
| Swin-Transformer backbone | **High** | ImageNet-22K pretrained weights available via HuggingFace/timm |
| Mask2Former decoder | **High** | Official PyTorch implementation from Meta AI Research available |
| Attention-Gated U-Net | **Moderate-High** | Well-studied; combine with Mask2Former pixel decoder |
| GNN skeletonization | **Moderate** | Requires custom graph construction from skeletons (scikit-image + PyG) |
| Pseudo-3D depth estimation | **Moderate** | Novel claim; based on Beer-Lambert law for translucent objects |

### 2.2 Resource Feasibility (Free Hardware Only 🆓)

> **You do NOT need a personal GPU.** The entire project can be executed using free cloud platforms with careful model selection.

#### Recommended Free Platforms

| Platform | GPU | VRAM | Limit | Use For |
|---|---|---|---|---|
| **Kaggle Notebooks** | P100 or T4 | 16GB / 15GB | 30 hrs/week | Main training — most reliable free option |
| **Google Colab Free** | T4 | 15GB | Session-based (~3–4 hrs) | Quick experiments, inference demo |
| **Google Colab Pro** | T4/V100 | 15–16GB | Priority access | If you have ₹999/month budget |
| **Lightning.ai (free)** | T4 | 15GB | 22 hrs/month free | Jupyter-like, persistent storage |
| **Paperspace Gradient** | M4000 | 8GB | Free tier | Lightweight model testing only |

#### Lightweight Model Substitutions (Designed for T4/P100)

| Original (High-End) | Free-Tier Replacement | VRAM | Drop in Quality |
|---|---|---|---|
| Swin-B (88M params) | **Swin-T (28M params)** | ~6GB | ~3–5% AP |
| Swin-L (197M params) | **Swin-S (50M params)** | ~9GB | ~1–2% AP |
| Mask2Former full | **Mask2Former-Small** | ~8GB | Minimal |
| Real-ESRGAN x4 | **Real-ESRGAN x2** (lighter) | ~3GB | Fine for 1024→2048 |
| 2048×2048 input | **1024×1024 input** | 4× less | Minimal for fibrils |

#### Practical Training Budget on Kaggle (Free)

```
Stage 1 — Real-ESRGAN fine-tune:  ~2–3 hrs  on T4  (small dataset)
Stage 2+3 — Swin-T + Mask2Former: ~6–10 hrs on P100 (batch_size=1, grad_accum=8)
Stage 4 — GNN on CPU:             ~30 min   (graphs are tiny)
Total per full training run:       ~12 hrs   → fits Kaggle's 30 hr/week limit
```

#### Memory Tricks to Fit in 15GB VRAM

```python
# 1. Mixed Precision (cuts VRAM by ~40%)
from torch.cuda.amp import autocast, GradScaler

# 2. Gradient Checkpointing (trade compute for memory)
model.backbone.gradient_checkpointing_enable()

# 3. Gradient Accumulation (simulate large batch)
# Instead of batch_size=8 (won't fit), use:
batch_size = 1
accumulation_steps = 8  # effective batch = 8

# 4. Frozen backbone for first 5 epochs (saves ~3GB)
for param in model.backbone.parameters():
    param.requires_grad = False

# 5. Input size: 512×512 for prototyping, 1024×1024 for final runs
```

- **Training Time (T4/P100):** 10–18 hours total for full pipeline
- **Dataset size needed:** 200–500 images (totally feasible on free tier)
- **Storage:** Kaggle gives 20GB free — more than enough

### 2.3 Timeline Feasibility (6-month capstone)

```
Month 1:  Literature review, dataset collection, annotation setup
Month 2:  Stage 1 (Real-ESRGAN fine-tuning) + Stage 2 (Swin backbone training)
Month 3:  Stage 3 (Mask2Former integration, instance segmentation training)
Month 4:  Stage 4 (Skeletonization + GNN pipeline)
Month 5:  Integration, ablation studies, metric computation
Month 6:  Paper writing, results visualization, submission
```

---

## 3. 📦 Dataset Strategy (Crucial — Most Underestimated Part)

### 3.1 The Hard Truth
There is **no large public benchmark dataset** specifically for labeled pulp fibril segmentation. You must build a hybrid strategy.

### 3.2 Recommended Dataset Sources

#### A. Collect Your Own (Primary, Most Publishable)
- **Equipment Needed:** Phase-contrast optical microscope or confocal laser scanning microscope
- Contact a **paper/pulp mill** or university materials lab for sample images
- Tools: ImageJ/FIJI for manual annotation → **LabelMe** or **CVAT** for polygon instance masks
- Target: **200–500 images** with full instance annotations is sufficient for transfer learning
- **This becomes your novel benchmark** — cite it as a contribution

#### B. Transfer Learning / Proxy Datasets
| Dataset | Link | Usage |
|---|---|---|
| **BCSS (Breast Cancer Semantic Seg)** | TCGA/NIH | Fiber-like cell topology; proxy pretraining |
| **NeuroMorpho / SWC files** | neuromorpho.org | Neurite/axon morphology — structurally similar to fibrils |
| **CompositeSegDB (carbon fiber SEM)** | Published as part of MatSci papers | Synthetic fiber overlap dataset |
| **Zenodo FIB-SEM datasets** | zenodo.org | Fiber-like EM structures for pretraining |
| **StarDist cell dataset** | github.com/stardist | Convex shape instances + irregular boundaries |

#### C. Synthetic Data Generation (Strong Research Contribution)
- Use **Blender** or **Python (scikit-image + PIL)** to generate synthetic fibril microscopy images
- Procedurally render:
  - Random curved fibers with realistic thickness (0.5–3μm at simulated scale)
  - Translucency via alpha compositing
  - Overlapping with realistic occlusion
  - Add Gaussian noise, blur, and low-contrast simulation
- Use **CycleGAN** or **SPADE** to convert synthetic → realistic domain
- Label: automatically known (ground truth is the generation parameters)
- Cite as: *"FibrilSynth: A Synthetic Benchmark for Fiber Instance Segmentation"*

#### D. Data Augmentation Strategy
```python
# Augmentation pipeline (Albumentations)
transforms = [
    RandomRotate90(),
    ElasticTransform(alpha=120, sigma=6),   # Simulate fiber bending
    GaussianNoise(var_limit=(5, 25)),        # Microscope noise
    CLAHE(clip_limit=2.0),                  # Contrast enhancement
    RandomBrightnessContrast(limit=0.3),    # Low-contrast simulation
    CoarseDropout(max_holes=8, max_height=32) # Simulated occlusion
]
```

---

## 4. 🧠 Detailed Architecture Design

### 4.1 Full Pipeline Diagram

```
[RAW IMAGE 1024×1024 Grayscale]
         │
         ▼
┌─────────────────────────────────────┐
│  STAGE 1: Real-ESRGAN Upsampler     │
│  Input: 1024×1024 → Output: 2048×2048│
│  Fine-tuned on microscopy domain    │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  STAGE 2: Swin-Transformer Backbone  │
│  Model: Swin-B or Swin-L            │
│  Pretrained: ImageNet-22K           │
│  Output: 4× hierarchical feature maps│
│  C1(192,512²), C2(384,256²),        │
│  C3(768,128²), C4(1536,64²)         │
└────────────────┬────────────────────┘
                 │
         ┌───────┴───────┐
         ▼               ▼
┌──────────────┐  ┌─────────────────────┐
│ FPN Pixel    │  │ Transformer Decoder  │
│ Decoder      │  │ (Mask2Former)        │
│ (Attn-UNet)  │  │ 100 Object Queries   │
│              │  │ Masked Cross-Attn    │
│ AG-Skip Conn │  │ → Per-query mask +   │
│ → Per-pixel  │  │   class score        │
│ embeddings   │  └──────────┬──────────┘
└──────┬───────┘             │
       └──────────┬──────────┘
                  ▼
┌─────────────────────────────────────┐
│  STAGE 3: Instance Mask Output      │
│  N × (H × W) binary masks          │
│  Each mask = one fibril instance    │
│  Color-coded visualization          │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  STAGE 4: Skeletonization + GNN     │
│  scikit-image.morphology.skeletonize│
│  → Graph: nodes=junction pts,       │
│    edges=fiber segments             │
│  GNN (GraphSAGE) → predict:         │
│  • Length (px → μm)                 │
│  • Branching angle                  │
│  • Tortuosity index                 │
│  • Pseudo-3D depth (intensity)      │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  OUTPUT: Metric Report (CSV/JSON)   │
│  • Fibrillation Index               │
│  • Per-fibril Length/Width          │
│  • Branching Count                  │
│  • Tortuosity (curliness)           │
│  • Estimated 3D depth               │
└─────────────────────────────────────┘
```

### 4.2 Stage 1 — Real-ESRGAN Super Resolution

**Why Real-ESRGAN over SRCNN/EDSR?**
- Real-ESRGAN uses a **U-Net discriminator with spectral normalization** — handles blind degradation (noise, blur, compression artifacts) simultaneously
- Trained on synthetic degradation pipelines matching real microscope imperfections

**Implementation:**
```python
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

model = RRDBNet(num_in_ch=1, num_out_ch=1, num_feat=64,  # Grayscale
                num_block=23, num_grow_ch=32, scale=2)
upsampler = RealESRGANer(scale=2, model_path='weights/RealESRGAN_x2.pth',
                          model=model, tile=512, tile_pad=10)

# Fine-tune on your microscopy pairs
```

**Innovation:** Fine-tune on synthetically degraded ↔ clean microscopy pairs (self-supervised). This is novel for pulp fiber domain.

### 4.3 Stage 2 — Swin-Transformer Backbone

**Why Swin over ViT?**

| Property | ViT | Swin Transformer |
|---|---|---|
| Attention scope | Global (all patches) | Local shifted windows → Global |
| Complexity | O(N²) | O(N) — linear |
| Multi-scale features | ❌ No | ✅ Yes (hierarchical stages) |
| Suitability for detection | Poor | Excellent |
| COCO instance seg SOTA | Below Swin | Swin-L: 58.7 AP |

**Swin Internal Mechanism:**
1. Image split into non-overlapping 4×4 patches → patch embeddings
2. **W-MSA** (Window Multi-head Self-Attention) within 7×7 windows
3. **SW-MSA** (Shifted Window MSA) — windows shift by (⌊M/2⌋, ⌊M/2⌋) to enable cross-window communication
4. **Patch Merging** between stages — analogous to CNN pooling, creates hierarchy

**For your fibril images:**
- Swin-B: 88M params — recommendable for capstone
- Swin-L: 197M params — use if Colab A100 available
- Input: Grayscale 2048×2048 (after SR) → resize to 1024×1024 for memory, or use window_size=16

### 4.4 Stage 3 — Mask2Former Decoder + Attention U-Net Pixel Decoder

**Mask2Former Core Innovation:**
Standard transformer decoders use full-image cross-attention → expensive + unfocused. Mask2Former uses **Masked Cross-Attention**:
- Each object query **only attends to its predicted foreground mask region**
- This constrains attention → faster convergence + better localization
- Result: Each query learns ONE fibril → natural instance separation

```
Query Q₁ → attends to [Fibril Region A] → predicts Mask A
Query Q₂ → attends to [Fibril Region B] → predicts Mask B
(even if A and B overlap, they are computed independently)
```

**Why this solves occlusion:**
Two crossing fibrils A and B have **different query assignments**. Each query's masked attention is initialized independently, so the model asks: "Given that I'm looking for Fibril A, where is it?" — ignoring Fibril B's pixels.

**Attention-Gated U-Net as Pixel Decoder:**
```
Replace standard FPN pixel decoder with AG-UNet:

Feature C4 (coarsest) → Attention Gate → ⊕ → Upsample
                    ↑                   ↑
Feature C3 ─────────────────────────────
                    ↑
Feature C2 → Attention Gate → ⊕ → Upsample
...

Attention Gate Formula:
  x̂ = x ⊙ σ(W_x·x + W_g·g + b)    [where g = gating signal from decoder]
  σ = sigmoid, ⊙ = element-wise mul
```

**Why attention gates:**
For fibrils, the relevant pixels are the thin, high-frequency edges. The attention gate learns to **upweight fibril-edge pixels and downweight background noise/water bubbles** — directly solving the low-contrast problem.

### 4.5 Stage 4 — Skeletonization + GNN Quantification

**Skeletonization Algorithm:**
```python
from skimage.morphology import skeletonize, thin
from skimage.measure import label, regionprops
import networkx as nx

def mask_to_graph(binary_mask):
    skeleton = skeletonize(binary_mask)
    # Find junction points (pixels with >2 neighbors) 
    # Build NetworkX graph: nodes=endpoints+junctions, edges=skeleton segments
    G = build_fibril_graph(skeleton)
    return G

def compute_metrics(G, px_to_um=0.25):  # calibrate per microscope
    metrics = {}
    for edge in G.edges():
        path = nx.shortest_path(G, edge[0], edge[1])
        length_um = len(path) * px_to_um
        # Tortuosity = path length / Euclidean distance
        euclidean = np.linalg.norm(np.array(edge[0]) - np.array(edge[1]))
        tortuosity = length_um / (euclidean * px_to_um)
        metrics[edge] = {'length': length_um, 'tortuosity': tortuosity}
    return metrics
```

**GNN for Metric Prediction:**
- Use **GraphSAGE** or **GAT (Graph Attention Network)**
- Node features: local pixel intensity, gradient magnitude, curvature at junction points
- Edge features: segment length (in pixels), mean intensity, std dev
- Task: predict per-edge quantitative metrics + graph-level Fibrillation Index
- This is publishable as a novel application of GNNs to fiber analysis

**Pseudo-3D Depth from Intensity:**
- Based on Beer-Lambert Law: $I = I_0 \cdot e^{-\mu \cdot d}$ 
  - Where $\mu$ = absorption coefficient, $d$ = depth through fiber
- Translucent fibers at different depth planes → different transmitted intensity
- By fitting intensity profiles to Beer-Lambert model along the skeleton path, estimate relative depth $d$ of each fibril
- **Novel contribution** — no paper has used this for pulp fibrils

---

## 5. 📐 Mathematical Formulations

### 5.1 Fibrillation Index (Your Primary Metric)
$$FI = \frac{P_{fiber+fibrils} - P_{fiber\_only}}{P_{fiber\_only}} \times 100\%$$
Where $P$ = perimeter measured from the instance mask in pixels, converted to μm.

### 5.2 Tortuosity Index
$$T = \frac{L_{path}}{L_{Euclidean}}$$
- $T = 1$: Perfectly straight fibril
- $T > 1$: Curly/branched (higher = more defibrillation)

### 5.3 Attention Gate (Additive)
$$\alpha_i = \sigma_2(\psi^T(\sigma_1(W_x^T x_i + W_g^T g_i + b_g)) + b_\psi)$$
$$\hat{x}_i = \alpha_i \odot x_i$$

### 5.4 Mask2Former Loss
$$\mathcal{L} = \lambda_{cls} \cdot \mathcal{L}_{cls} + \lambda_{mask} \cdot \mathcal{L}_{mask} + \lambda_{dice} \cdot \mathcal{L}_{dice}$$
- $\mathcal{L}_{cls}$: Focal loss for query-class assignment
- $\mathcal{L}_{mask}$: Binary cross-entropy on predicted mask vs. GT
- $\mathcal{L}_{dice}$: Dice loss for mask overlap quality

### 5.5 Hungarian Matching (Key to Why Mask2Former Works)
During training, predicted queries are matched to ground-truth instances using:
$$\hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_N} \sum_{i=1}^N \mathcal{C}_{match}(p_i, \hat{p}_{\sigma(i)})$$
This ensures each GT fibril is matched to exactly one predicted query → prevents double-counting.

---

## 6. 📊 Evaluation Metrics

### 6.1 Segmentation Metrics
| Metric | Formula | Target |
|---|---|---|
| **AP (Average Precision)** | Area under PR curve at IoU 0.5:0.95 | > 50% |
| **AP50** | Precision at IoU ≥ 0.5 | > 70% |
| **AP_small** | AP for objects < 32² px | Key for tiny fibrils |
| **PQ (Panoptic Quality)** | SQ × RQ | > 0.55 |
| **Dice Coefficient** | 2×(A∩B)/(A+B) | > 0.80 |
| **mIoU** | Mean per-class IoU | > 0.75 |

### 6.2 Quantification Accuracy Metrics (Novel)
| Metric | Method |
|---|---|
| **Length RMSE** | Compare predicted skeleton length to manually measured ground truth |
| **Fibrillation Index MAE** | vs. Kajaani FiberLab / Morfi neo commercial analyzer |
| **Branching Point Detection F1** | Precision/Recall at detected junction nodes |

### 6.3 Baseline Comparisons (Must Have for Paper)
| Baseline | Why Include |
|---|---|
| U-Net (vanilla) | Classic semantic seg baseline |
| Mask R-CNN | Standard instance seg baseline |
| SAM (Segment Anything) | SOTA zero-shot, shows your method beats it on this domain |
| PointRend | Strong boundary prediction |
| FiberNet (if accessible) | Domain-specific prior art |

---

## 7. 🔧 Training Strategy

### 7.1 Three-Phase Training Protocol

**Phase 1 (Warmup — 5 epochs):**
- Freeze Swin backbone, train only Mask2Former decoder heads
- LR: 1e-4 for decoder, backbone frozen
- Purpose: Initialize decoder weights without destroying pretrained backbone

**Phase 2 (Joint Finetuning — 30 epochs):**
- Unfreeze full model
- LR: 1e-5 (backbone) + 1e-4 (decoder) — differential learning rates
- Optimizer: AdamW, weight decay = 0.05
- Scheduler: Cosine annealing with warm restarts

**Phase 3 (Real-ESRGAN fine-tuning — separate):**
- Self-supervised: degrade clean images with synthetic blur+noise → train ESRGAN to reconstruct
- Loss: Perceptual loss (VGG-based) + adversarial loss + L1
- 10 epochs on ~500 pairs is sufficient

### 7.2 Key Hyperparameters (Tuned for Free Tier — T4/P100)
```python
config = {
    "backbone": "swin_tiny_patch4_window7_224",  # 28M params, fits T4 easily
    # Upgrade to swin_small_... if Kaggle P100 available
    "input_size": 512,           # Use 1024 only for final run on P100
    "num_queries": 50,           # Reduce from 100 → fibrils < 50 per image
    "hidden_dim": 256,
    "nheads": 8,
    "enc_layers": 6,
    "dec_layers": 6,             # Reduce from 9 → saves ~1.5GB
    "batch_size": 1,             # Only 1 image at a time on T4
    "accum_grad_steps": 8,       # Effective batch size = 8
    "lr_backbone": 1e-5,
    "lr_head": 1e-4,
    "weight_decay": 0.05,
    "epochs": 30,                # Fewer epochs, same result with pretrained backbone
    "mixed_precision": True,     # MANDATORY on free tier
    "gradient_checkpointing": True,  # Saves ~30% VRAM
    "loss_weights": {"cls": 2.0, "mask": 5.0, "dice": 5.0}
}
```

### 7.3 Mixed Precision + Memory Tricks
```python
# Use torch.cuda.amp for memory savings
scaler = torch.cuda.amp.GradScaler()
with torch.autocast(device_type='cuda'):
    outputs = model(images)
    loss = criterion(outputs, targets)
scaler.scale(loss).backward()
scaler.step(optimizer)
```

---

## 8. 💡 Novelty Claims (For Your Research Paper)

This is critical for a publishable paper. Here are your **5 strong novelty arguments:**

### Claim 1: First domain-specific SR-augmented fibril segmentation pipeline
> *"We are the first to employ Real-ESRGAN as a preprocessing stage specifically tuned for pulp fibril microscopy, demonstrating X% improvement in downstream segmentation AP compared to direct processing of raw images."*

### Claim 2: First Swin-Transformer + Mask2Former application to industrial fiber analysis
> *"While Mask2Former has been applied to natural image segmentation, we present the first application to semi-transparent industrial fiber microscopy, addressing the unique challenges of translucency and sub-micron scale variability."*

### Claim 3: Attention-gated pixel decoder for low-contrast fibril highlighting
> *"We replace the standard FPN pixel decoder in Mask2Former with an Attention-Gated U-Net decoder, specifically designed to suppress noise-dominated background and amplify high-frequency fibril-edge responses in low-SNR microscopy images."*

### Claim 4: GNN-based morphological quantification with pseudo-3D estimation
> *"We introduce a skeleton-graph-to-GNN pipeline that outputs a structured Fibrillation Report, including tortuosity, branching angle distribution, and a novel pseudo-3D depth estimate derived from Beer-Lambert intensity modeling of translucent fibers."*

### Claim 5: Novel FibrilBench synthetic dataset
> *"We release FibrilSynth, a procedurally generated annotated dataset of 5000 simulated pulp fibril microscopy images with pixel-perfect ground truth, enabling fair evaluation of future methods."*

---

## 9. 🛠️ Implementation Stack

```
Language:    Python 3.10
Framework:   PyTorch 2.1
Models:      
  - Swin-Transformer: timm / microsoft/swin-base-patch4-window12-384
  - Mask2Former: facebookresearch/Mask2Former (official)  
  - Real-ESRGAN: xinntao/Real-ESRGAN
  - Attention U-Net: custom or segmentation_models_pytorch
Annotation:  CVAT (cvat.ai) — free, supports COCO polygon format
Skeletonize: scikit-image 0.21
Graph:       NetworkX + PyTorch Geometric (GraphSAGE/GAT)
Tracking:    Weights & Biases (wandb)
Serving:     Gradio (for demo UI)
Metrics:     pycocotools (COCO AP), custom FI calculator
```

### 9.1 Key Code Files Structure
```
fibril_seg/
├── data/
│   ├── dataset.py          # FibrilDataset class (COCO format)
│   ├── augmentation.py     # Albumentations pipeline
│   └── synthetic_gen.py    # FibrilSynth generator
├── models/
│   ├── esrgan_sr.py        # Stage 1: SR module
│   ├── swin_backbone.py    # Stage 2: Swin-T feature extractor
│   ├── ag_pixel_decoder.py # AG-UNet pixel decoder
│   ├── mask2former.py      # Stage 3: Full Mask2Former
│   └── gnn_quantifier.py   # Stage 4: GraphSAGE quantifier
├── training/
│   ├── train.py            # Main training loop
│   ├── losses.py           # Focal + Dice + CE losses
│   └── hungarian_matcher.py# Bipartite matching
├── evaluation/
│   ├── coco_eval.py        # AP metrics
│   └── fibril_metrics.py   # FI, tortuosity, branching
└── inference/
    ├── predict.py          # Single image inference
    └── report_generator.py # CSV/JSON report output
```

---

## 10. 📰 Research Paper Structure (How to Write It)

### Recommended Venue Targets
| Venue | Type | Impact | Deadline |
|---|---|---|---|
| **IEEE TPAMI** | Journal | IF ~24 | Rolling |
| **Pattern Recognition** | Journal | IF ~8 | Rolling |
| **CVPR Workshop (BioImage)** | Conference | Prestigious | Jan/Feb |
| **MICCAI** | Conference | IF equivalent | Feb/Mar |
| **Nordic Pulp & Paper Research Journal** | Domain journal | Specialized | Rolling |
| **Computers & Electronics in Agriculture** | Crossover | IF ~6 | Rolling |

### Paper Outline
```
1. Abstract (250 words)
2. Introduction
   - Industrial motivation (energy waste in over-refining)
   - Technical problem statement
   - Contributions (bulleted list of 4-5 claims)
3. Related Work
   - Fiber analysis methods (pre-DL vs. DL)
   - Instance segmentation evolution (Mask R-CNN → Mask2Former)
   - Swin Transformers in microscopy
4. Methodology
   4.1 Dataset & Annotations
   4.2 SR Preprocessing (Real-ESRGAN)
   4.3 Swin-T Backbone
   4.4 AG-UNet Pixel Decoder
   4.5 Mask2Former Instance Head
   4.6 Skeletonization + GNN Quantifier
5. Experiments
   5.1 Implementation Details
   5.2 Ablation Studies (each component's contribution)
   5.3 Comparison with Baselines
   5.4 Quantification Accuracy
6. Results & Discussion
7. Conclusion
```

### Critical Ablation Studies (Must Include)
| Experiment | What You Prove |
|---|---|
| w/o SR preprocessing | SR improves AP by X% |
| Swin-B vs. ResNet-50 backbone | Transformer backbone is better |
| Standard FPN vs. AG-UNet decoder | Attention gates help low-contrast |
| Mask2Former vs. Mask R-CNN | Query-based is better for occlusion |
| w/o GNN (just skeleton metrics) | GNN adds quantification accuracy |

---

## 11. ⚠️ Risks and Mitigation Strategies

| Risk | Probability | Mitigation |
|---|---|---|
| Not enough real annotated images | High | Use FibrilSynth + transfer learning |
| Swin-L too large for available GPU | Medium | Use Swin-B; gradient checkpointing |
| No collaborating pulp mill | Medium | Use published paper images + digital staining |
| Pseudo-3D estimation underwhelms | Medium | Frame as "preliminary 3D analysis" with Beer-Lambert justification |
| GNN training unstable | Low | Start with classic skeleton metrics; GNN as optional enhancement |
| Mask2Former official code incompatible | Low | Port to detectron2 or use unofficial HuggingFace version |

---

## 12. 🚀 What Makes This Research Excellent

Your project hits **6 key innovation axes** that make for strong, publishable research:

1. **🏭 Industrial Relevance** — Directly tied to energy savings (US$millions/year scale)
2. **🧠 Novel Architecture** — Specific combination of SR+Swin+AG-UNet+Mask2Former is unprecedented
3. **📊 Novel Dataset** — FibrilSynth benchmark is itself a contribution
4. **🔢 Quantitative Output** — Goes beyond "here's a mask" to actionable industrial metrics
5. **📐 Physical Modeling** — Beer-Lambert pseudo-3D is a physics-informed ML approach
6. **🔗 Cross-disciplinary** — Bridges materials science + DL + graph theory

> **Bottom line:** This is not a "apply SOTA model X to domain Y" project. You are building a full pipeline with novel components at each stage, targeting a real industrial pain point. If executed well, this can be published in an IEEE-tier journal or top workshop.

---

## 13. 🔑 Key References to Cite

1. **Mask2Former:** Cheng et al., "Masked-attention Mask Transformer for Universal Image Segmentation," CVPR 2022
2. **Swin-T:** Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows," ICCV 2021
3. **Attention U-Net:** Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas," MIDL 2018
4. **Real-ESRGAN:** Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data," ICCV 2021
5. **ESRGAN for DHM:** Published on arXiv — Resolution enhancement in digital holographic microscopy
6. **FiberNet:** (ResearchGate) DCNN for multi-fiber instance segmentation in microscopy
7. **BCNet:** Ke et al., "Deep Occlusion-Aware Instance Segmentation with Overlapping BiLayers," CVPR 2021
8. **Valmet Fiber Analyzer:** Industrial benchmarking system for fibrillation index measurement
9. **StarDist:** Schmidt et al., "Cell Detection with Star-convex Polygons," MICCAI 2018 (useful comparison)
10. **GraphSAGE:** Hamilton et al., "Inductive Representation Learning on Large Graphs," NeurIPS 2017
