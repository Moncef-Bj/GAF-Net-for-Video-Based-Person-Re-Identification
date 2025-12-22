#!/usr/bin/env python3
"""
GAF-Net: Gait and Appearance Features Fusion for Video-Based Person Re-Identification

Paper: VISAPP 2024

Fusion Formula:
    f_fused = concat(normalize(f_app), λ * normalize(f_gait))

Results on iLIDS-VID:
    PiT + Gait1:   93.07% Rank-1 @ λ=0.74
    MGH + Gait2:   90.40% Rank-1 @ λ=0.84
    OSNet + Gait1: 70.93% Rank-1 @ λ=0.90

Usage:
    python evaluate.py
    python evaluate.py --backbone pit
    python evaluate.py --backbone mgh --lambda_val 0.84
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from pathlib import Path


# =============================================================================
# Configuration
# =============================================================================

BACKBONES = {
    'pit': {
        'name': 'PiT (Pooling-based Vision Transformer)',
        'appearance_dim': 9216,
        'optimal_lambda': 0.74,
        'paper_rank1': 93.07,
        'gait_source': 'gait1',
    },
    'mgh': {
        'name': 'MGH (Multi-Granularity Hypergraph)',
        'appearance_dim': 5120,
        'optimal_lambda': 0.84,
        'paper_rank1': 90.40,
        'gait_source': 'gait2',
    },
    'osnet': {
        'name': 'OSNet (Omni-Scale Network)',
        'appearance_dim': 512,
        'optimal_lambda': 0.90,
        'paper_rank1': 70.93,
        'gait_source': 'gait1',
    },
}

GAIT_DIM = 128
NUM_SPLITS = 10


# =============================================================================
# Evaluation Functions
# =============================================================================

def compute_distance_matrix(query_feat, gallery_feat):
    """Compute squared Euclidean distance after L2 normalization."""
    qf = F.normalize(torch.from_numpy(query_feat), dim=1, p=2)
    gf = F.normalize(torch.from_numpy(gallery_feat), dim=1, p=2)
    
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    
    return distmat.numpy()


def evaluate_cmc(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Compute CMC curve and mAP."""
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    num_valid_q = 0.

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def load_appearance(filepath, app_dim):
    """Load appearance embeddings from CSV."""
    data = pd.read_csv(filepath).to_numpy()
    embeddings = data[:, 1:app_dim]
    pids = data[:, app_dim].astype(int)
    camids = data[:, app_dim + 1].astype(int)
    return embeddings, pids, camids


def load_gait(filepath):
    """Load gait embeddings from CSV."""
    data = pd.read_csv(filepath).to_numpy()
    embeddings = data[:, :GAIT_DIM]
    return embeddings


def evaluate_backbone(data_path, backbone, lambda_val=None):
    """Evaluate a backbone across all splits."""
    config = BACKBONES[backbone]
    data_path = Path(data_path)
    
    if lambda_val is None:
        lambda_val = config['optimal_lambda']
    
    app_dim = config['appearance_dim']
    gait_source = config['gait_source']
    
    print(f"\n{'='*60}")
    print(f"Evaluating {config['name']}")
    print(f"  Appearance dim: {app_dim}")
    print(f"  Gait source: {gait_source}")
    print(f"  λ = {lambda_val}")
    print(f"{'='*60}")
    
    cmcs = []
    mAPs = []
    
    for i in range(NUM_SPLITS):
        # Load appearance
        g_app, g_pids, g_camids = load_appearance(
            data_path / 'ilids' / backbone / f'gallery_split{i}.csv', app_dim
        )
        q_app, q_pids, q_camids = load_appearance(
            data_path / 'ilids' / backbone / f'query_split{i}.csv', app_dim
        )
        
        # Load gait
        g_gait = load_gait(data_path / 'ilids' / gait_source / f'gallery_split{i}.csv')
        q_gait = load_gait(data_path / 'ilids' / gait_source / f'query_split{i}.csv')
        
        # Fusion: concat(normalize(app), λ * normalize(gait))
        gallery_fused = np.hstack((normalize(g_app), lambda_val * normalize(g_gait)))
        query_fused = np.hstack((normalize(q_app), lambda_val * normalize(q_gait)))
        
        # Evaluate
        distmat = compute_distance_matrix(query_fused, gallery_fused)
        cmc, mAP = evaluate_cmc(distmat, q_pids, g_pids, q_camids, g_camids)
        
        cmcs.append(cmc)
        mAPs.append(mAP)
        
        print(f"  Split {i}: Rank-1 = {cmc[0]*100:.2f}%, mAP = {mAP*100:.2f}%")
    
    mean_cmc = np.stack(cmcs).mean(axis=0)
    mean_mAP = np.mean(mAPs)
    
    print(f"\n  Mean Rank-1: {mean_cmc[0]*100:.2f}%")
    print(f"  Mean Rank-5: {mean_cmc[4]*100:.2f}%")
    print(f"  Mean Rank-10: {mean_cmc[9]*100:.2f}%")
    print(f"  Mean mAP: {mean_mAP*100:.2f}%")
    print(f"  Paper Rank-1: {config['paper_rank1']}%")
    diff = mean_cmc[0]*100 - config['paper_rank1']
    print(f"  Difference: {diff:+.2f}%")
    
    return mean_cmc, mean_mAP


def main():
    parser = argparse.ArgumentParser(
        description='GAF-Net Evaluation on iLIDS-VID',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--data_path', type=str, default='./embeddings',
                        help='Path to embeddings directory')
    parser.add_argument('--backbone', type=str, choices=['pit', 'mgh', 'osnet', 'all'],
                        default='all', help='Backbone to evaluate')
    parser.add_argument('--lambda_val', type=float, default=None,
                        help='Fusion weight (uses paper optimal if not specified)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("GAF-Net: Gait and Appearance Features Fusion")
    print("Video-Based Person Re-Identification on iLIDS-VID")
    print("=" * 70)
    print(f"\nFormula: concat(normalize(app), λ * normalize(gait))")
    
    backbones = ['pit', 'mgh', 'osnet'] if args.backbone == 'all' else [args.backbone]
    results = {}
    
    for backbone in backbones:
        config = BACKBONES[backbone]
        lambda_val = args.lambda_val if args.lambda_val is not None else config['optimal_lambda']
        
        mean_cmc, mean_mAP = evaluate_backbone(args.data_path, backbone, lambda_val)
        
        results[backbone] = {
            'rank1': mean_cmc[0] * 100,
            'rank5': mean_cmc[4] * 100,
            'rank10': mean_cmc[9] * 100,
            'mAP': mean_mAP * 100,
            'lambda': lambda_val,
        }
    
    # Print summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Backbone':<10} {'Gait':<8} {'λ':<6} {'Rank-1':<10} {'Rank-5':<10} {'mAP':<10} {'Paper':<10} {'Match'}")
    print("-" * 78)
    
    for backbone in backbones:
        res = results[backbone]
        config = BACKBONES[backbone]
        paper = config['paper_rank1']
        gait = config['gait_source']
        diff = abs(res['rank1'] - paper)
        match = "✓" if diff < 0.5 else "✗"
        
        print(f"{backbone.upper():<10} {gait:<8} {res['lambda']:<6.2f} {res['rank1']:<10.2f} "
              f"{res['rank5']:<10.2f} {res['mAP']:<10.2f} {paper:<10.2f} {match}")
    
    print("=" * 70)


if __name__ == '__main__':
    main()
