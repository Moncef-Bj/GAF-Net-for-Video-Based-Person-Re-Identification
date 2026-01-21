# GAF-Net: Video-Based Person Re-Identification via Appearance and Gait Recognitions

[![Paper](https://img.shields.io/badge/Paper-VISAPP%202024-blue)](https://doi.org/10.5220/0012364200003660)
[![HAL](https://img.shields.io/badge/HAL-hal--04524979-orange)](https://amu.hal.science/hal-04524979)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red)](https://pytorch.org)

Official implementation of **GAF-Net: Video-Based Person Re-Identification via Appearance and Gait Recognitions** (VISAPP 2024).

**Authors:** Moncef Boujou, Rabah Iguernaissi, Lionel Nicod, Djamal Merad, Séverine Dubuisson

**Affiliations:** LIS, CNRS, Aix-Marseille University, France | CERGAM, Aix-Marseille University, France

##  Abstract

Video-based person re-identification (Re-ID) is a challenging task aiming to match individuals across various cameras based on video sequences. While most existing Re-ID techniques focus solely on appearance information, including gait information could potentially improve person Re-ID systems. In this study, we propose GAF-Net, a novel approach that integrates appearance with gait features for re-identifying individuals; the appearance features are extracted from RGB tracklets while the gait features are extracted from skeletal pose estimation. These features are then combined into a single feature allowing the re-identification of individuals.


**This repository provides:**
- Evaluation script to reproduce paper results
- Pre-computed embeddings (appearance + gait)
- GaitGraph training code adapted for iLIDS-VID
- Pose data for training

**This repository does not provide:**
- Appearance model training (PiT/MGH/OSNet) — use original repos
- End-to-end inference pipeline (video → Re-ID)
- Pre-trained GaitGraph weights (coming soon)

##  Results on iLIDS-VID

| Method | Gait | Rank-1 | Rank-5 | Rank-10 | Rank-20 | λ |
|--------|------|--------|--------|---------|---------|---|
| **GAF-Net (PiT)** | gait1 | **93.07%** | 99.27% | 99.74% | 99.94% | 0.74 |
| **GAF-Net (MGH)** | gait2 | **90.40%** | 98.66% | 98.99% | 99.66% | 0.84 |
| **GAF-Net (OSNet)** | gait1 | **70.93%** | 88.40% | 93.00% | 96.54% | 0.90 |

### Improvement over Appearance-Only Baselines

| Backbone | Appearance Only | GAF-Net (+ Gait) | Improvement |
|----------|-----------------|------------------|-------------|
| PiT | 92.07% | **93.07%** | +1.00% |
| MGH | 85.60% | **90.40%** | +4.80% |
| OSNet | 59.20% | **70.93%** | +11.73% |

##  Installation

```bash
# Clone the repository
git clone https://github.com/Moncef-Bj/GAF-Net-for-Video-Based-Person-Re-Identification.git
cd GAF-Net-for-Video-Based-Person-Re-Identification

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python >= 3.8
- PyTorch >= 1.9
- NumPy
- Pandas
- scikit-learn
- torchreid

##  Download Data

Download the pre-computed embeddings and pose data from Google Drive:

**[ Download gaf-net-data.zip (395 MB)](https://drive.google.com/file/d/1oop2lyLT3nceZMtPrreMjY2yUv9bGea3/view?usp=drive_link)**

Extract the zip in the repository root:

```bash
# Linux/Mac
unzip gaf-net-data.zip -d .

# Windows (PowerShell)
Expand-Archive -Path gaf-net-data.zip -DestinationPath .
```

This will create:
```
GAF-Net-for-Video-Based-Person-Re-Identification/
├── embeddings/           # Pre-computed embeddings for evaluation
│   └── ilids/
│       ├── pit/          # PiT appearance embeddings (9216-d)
│       ├── mgh/          # MGH appearance embeddings (5120-d)
│       ├── osnet/        # OSNet appearance embeddings (512-d)
│       ├── gait1/        # GaitGraph embeddings for PiT & OSNet (128-d)
│       └── gait2/        # GaitGraph embeddings for MGH (128-d)
└── poses/                # Pose data for training GaitGraph
    ├── splits_pit/       # Poses for PiT & OSNet splits
    └── splits_mgh/       # Poses for MGH splits
```

##  Repository Structure

```
GAF-Net-for-Video-Based-Person-Re-Identification/
├── README.md
├── requirements.txt
├── evaluate.py                 # Main evaluation script (reproduces paper results)
├── embeddings/                 # (download from Google Drive)
├── poses/                      # (download from Google Drive)
└── gaitgraph/                  # Modified GaitGraph code
    └── src/
        ├── train.py            # Training script
        ├── evaluate.py         # Evaluation & embedding extraction
        ├── common.py           # Configuration & model setup
        ├── losses.py           # SupConLoss
        └── datasets/
            ├── gait.py         # Dataset classes (iLIDS, MARS, CASIA-B)
            ├── augmentation.py # Data augmentation
            └── graph.py        # COCO skeleton graph
```

##  Quick Start

### Reproduce Paper Results

```bash
# Download and extract data first (see above)

# Evaluate all backbones
python evaluate.py

# Evaluate specific backbone
python evaluate.py --backbone pit

# Custom lambda value
python evaluate.py --backbone mgh --lambda_val 0.84
```

Expected output:
```
======================================================================
FINAL SUMMARY
======================================================================
Backbone   Gait     λ      Rank-1     Rank-5     mAP        Paper     
----------------------------------------------------------------------
PIT        gait1    0.74   93.07      99.33      95.80      93.07      
MGH        gait2    0.84   90.40      98.67      94.01      90.40      
OSNET      gait1    0.90   70.93      89.60      79.14      70.93      
======================================================================
```

##  Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'torchreid'**
```bash
pip install torchreid
# or
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

**2. FileNotFoundError: embeddings/ilids/...**
```
Make sure you downloaded and extracted gaf-net-data.zip in the repository root.
```

**3. Results don't match paper exactly**
```
Ensure you're using the correct lambda values:
- PiT: λ = 0.74
- MGH: λ = 0.84
- OSNet: λ = 0.90
```

##  Method

### Fusion Strategy

Our fusion formula combines appearance and gait modalities (Equation 4 in the paper):

```
z_i = [normalize(z_appearance), λ · normalize(z_gait)]
```

Where:
- `z_appearance`: Appearance embedding from PiT/MGH/OSNet
- `z_gait`: Gait embedding from GaitGraph (128-d)
- `λ`: Fusion weight, λ ∈ [0.6, 0.8] recommended
- `normalize()`: L2 normalization

### Architecture Overview

```
Video Sequence
      │
      ├──────────────────────────────────────┐
      │                                      │
      ▼                                      ▼
┌─────────────────┐              ┌─────────────────┐
│  Pose Estimator │              │  Appearance     │
│  (YOLO-Pose)    │              │ (PiT/MGH/OSNet) │
└─────────────────┘              └─────────────────┘
      │                                      │
      ▼                                      ▼
┌─────────────────┐              ┌─────────────────┐
│  Pose Sequence  │              │   Appearance    │
│  (T × 17 × 3)   │              │    Embedding    │
└─────────────────┘              └─────────────────┘
      │                                      │
      ▼                                      │
┌─────────────────┐                          │
│   GaitGraph     │                          │
│   (ResGCN)      │                          │
└─────────────────┘                          │
      │                                      │
      ▼                                      │
┌─────────────────┐                          │
│ Gait Embedding  │                          │
│    (128-d)      │                          │
└─────────────────┘                          │
      │                                      │
      └──────────────┬───────────────────────┘
                     ▼
            ┌────────────────┐
            │     Fusion     │
            │ [z_app, λ·z_g] │
            └────────────────┘
                     │
                     ▼
            ┌────────────────┐
            │ Final Embedding│
            └────────────────┘
```

##  Data Format

### Pose CSV Format (for GaitGraph training)

The pose files in `poses/` contain 2D pose estimations in COCO format (17 keypoints):

```csv
image_name,nose_x,nose_y,nose_conf,left_eye_x,left_eye_y,left_eye_conf,...,right_ankle_x,right_ankle_y,right_ankle_conf
cam1_person001_00319.png,30.04,22.35,0.278,29.14,20.0,0.046,...,20.52,124.65,0.457
```

**Format:**
- **Column 0**: Image filename (`cam{1|2}_person{XXX}_{FRAME}.png`)
- **Columns 1-51**: 17 COCO keypoints × 3 values (x, y, confidence) = 51 values

**COCO 17 Keypoints Order:**
```
0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
```

### Embedding CSV Format

**Appearance Embeddings** (`pit/`, `mgh/`, `osnet/`):
```csv
index,feat_0,feat_1,...,feat_N,person_id,camera_id
0,0.123,0.456,...,0.789,43,2
```

**Gait Embeddings** (`gait1/`, `gait2/`):
```csv
feat_0,feat_1,...,feat_127,person_id,camera_id
0.123,0.456,...,0.789,43,2
```

##  Training GaitGraph

### 1. Train GaitGraph

Our GaitGraph model follows a two-stage training approach:

**Stage 1: Pre-train on CASIA-B**
```bash
cd gaitgraph/src

python train.py casia-b /path/to/casia_b_train.csv \
    --valid_data_path /path/to/casia_b_test.csv \
    --batch_size 128 \
    --epochs 1000 \
    --learning_rate 1e-2 \
    --temp 0.01 \
    --sequence_length 60 \
    --network_name resgcn-n39-r8
```

**Stage 2: Fine-tune on iLIDS-VID**
```bash
# For PiT/OSNet splits
python train.py iLIDS ../poses/splits_pit/train_split0.csv \
    --valid_data_path ../poses/splits_pit/test_split0.csv \
    --batch_size 128 \
    --epochs 300 \
    --learning_rate 1e-5 \
    --temp 0.01 \
    --sequence_length 60 \
    --network_name resgcn-n39-r8 \
    --weight_path /path/to/casia_b_pretrained.pth

# For MGH splits
python train.py iLIDS ../poses/splits_mgh/train_split0.csv \
    --valid_data_path ../poses/splits_mgh/test_split0.csv \
    ...
```

### 2. Extract Gait Embeddings

After training, extract embeddings for fusion:

```bash
python evaluate.py iLIDS /path/to/test_split0.csv \
    --weight_path /path/to/ckpt_epoch_best.pth
```

##  Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{boujou2024gafnet,
  title={GAF-Net: Video-Based Person Re-Identification via Appearance and Gait Recognitions},
  author={Boujou, Moncef and Iguernaissi, Rabah and Nicod, Lionel and Merad, Djamal and Dubuisson, S{\'e}verine},
  booktitle={Proceedings of the 19th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISAPP)},
  pages={493--500},
  year={2024},
  organization={SCITEPRESS},
  doi={10.5220/0012364200003660}
}
```

##  Acknowledgements

This work builds upon:

### Appearance Models
- [PiT](https://git.openi.org.cn/zangxh/PiT.git) - Multidirection and Multiscale Pyramid in Transformer (Zang et al., IEEE TII 2022)
- [MGH](https://github.com/daodaofr/hypergraph_reid) - Multi-Granularity Hypergraph (Yan et al., CVPR 2020)
- [OSNet](https://github.com/KaiyangZhou/deep-person-reid) - Omni-Scale Network (Zhou et al., ICCV 2019)

### Gait Recognition
- [GaitGraph](https://github.com/tteepe/GaitGraph) - Graph Convolutional Network for Skeleton-Based Gait Recognition (Teepe et al., ICIP 2021)

### Tools & Libraries
- [YOLO-Pose](https://github.com/TexasInstruments/edgeai-yolov5) - Multi-Person Pose Estimation
- [torchreid](https://github.com/KaiyangZhou/deep-person-reid) - Person Re-Identification Library

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


##  Contact

For questions or issues, please open an issue or contact:
- Moncef Boujou - moncef.boujou@univ-amu.fr
