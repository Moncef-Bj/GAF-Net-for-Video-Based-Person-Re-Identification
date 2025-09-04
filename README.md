# GAF-Net: Video-Based Person Re-Identification via Appearance and Gait Recognitions

[![Paper](https://img.shields.io/badge/Paper-VISAPP%202024-blue)](https://www.scitepress.org/Link.aspx?doi=10.5220/0012364200003660)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Implementation of "GAF-Net: Video-Based Person Re-Identification via Appearance and Gait Recognitions" presented at VISAPP 2024.

## Overview

GAF-Net combines skeleton-based gait features with appearance features for video-based person re-identification. This is the first approach to integrate skeletal gait information (rather than silhouettes) with appearance data for Re-ID.

## Methodology

### Pipeline Overview
Our main contribution is the novel fusion of skeleton-based gait features with appearance features for person re-identification.

**Phase 1: Pose Extraction**


- OpenPose for 2D human pose estimation
- 17 keypoints per frame (COCO format)

**Phase 2: Feature Extraction**


- **Gait features**: Using GaitGraph approach [Teepe et al., 2021]
- **Appearance features**: Pre-trained models:
  - OSNet [Zhou et al., 2019] 
  - MGH [Yan et al., 2020]
  - PiT [Zang et al., 2022]

**Phase 3: Fusion **


- Weighted concatenation of skeleton-based gait + appearance features
- First approach to use skeleton-based (not silhouette) gait with appearance
- Systematic evaluation of different fusion strategies

### Implementation Approach

This work follows a modular pipeline where each component is trained/processed separately before fusion, allowing for systematic ablation studies as presented in our paper.

### Key Features
- Novel skeleton-based gait feature extraction using pose estimation
- Integration with multiple appearance backbones: OSNet, MGH, PiT  
- Weighted feature fusion approach
- Evaluation on iLIDS-VID dataset

## Results

| Method | Dataset | Rank-1 | Rank-5 | 
|--------|---------|--------|--------|
| GAF-Net (PiT) | iLIDS-VID | **93.07%** | **99.27%** |
| GAF-Net (MGH) | iLIDS-VID | 90.40% | 98.66% |
| GAF-Net (OSNet) | iLIDS-VID | 70.93% | 88.40% |

## Quick Start

```bash
# Clone repository
git clone https://github.com/Moncef-Bj/GAF-Net-for-Video-Based-Person-Re-Identification.git
cd GAF-Net-for-Video-Based-Person-Re-Identification

# Setup environment (coming soon)
pip install -r requirements.txt
```
## Repository Status

🚧 **Currently under development**

- [x] VISAPP 2024 paper published  
- [x] Core methodology validated
- [ ] Implementation in progress
- [ ] Training scripts
- [ ] Pre-trained models
- [ ] Evaluation code

## Citation

```bibtex
@conference{boujou2024gafnet,
  title={GAF-Net: Video-Based Person Re-Identification via Appearance and Gait Recognitions},
  author={Boujou, Moncef and Iguernaissi, Rabah and Nicod, Lionel and Merad, Djamal and Dubuisson, Séverine},
  booktitle={Proceedings of the 19th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications - Volume 4: VISAPP},
  year={2024},
  pages={870-877},
  publisher={SciTePress}
}
```
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
