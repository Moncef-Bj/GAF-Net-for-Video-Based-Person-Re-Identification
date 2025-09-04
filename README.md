# GAF-Net: Video-Based Person Re-Identification via Appearance and Gait Recognitions

[![Paper](https://img.shields.io/badge/Paper-VISAPP%202024-blue)](https://www.scitepress.org/Papers/2024/120749/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Implementation of "GAF-Net: Video-Based Person Re-Identification via Appearance and Gait Recognitions" presented at VISAPP 2024.

## Overview

GAF-Net combines skeleton-based gait features with appearance features for video-based person re-identification. This is the first approach to integrate skeletal gait information (rather than silhouettes) with appearance data for Re-ID.

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