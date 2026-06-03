'''
    Render CityStreet voxel predictions as an interactive 3D Plotly scene.
'''

import argparse
import json
import os
from os import path as osp

os.environ.setdefault('PRETRAINED', '0')
os.environ.setdefault('USE_MSDA_PYTORCH', '1')
os.environ.setdefault('DISABLE_TENSORBOARD', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict

from projects import Config, DATASET, HELPERS, MODELS


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize CountFormer 3D voxel output')
    parser.add_argument('--cfg_path', default='projects/configs/citystreet/citystreet_voxel_kaggle_tiny.py')
    parser.add_argument('--chkp_path', default=None)
    parser.add_argument('--part', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--sample_idx', type=int, default=0)
    parser.add_argument('--topk', type=int, default=1200)
    parser.add_argument('--out', default='workdirs/kaggle_3d_visualization.html')
    return parser.parse_args()


def optional_arg(value):
    return value if value else None


def find_checkpoint():
    run_root = 'workdirs/citystreet/vox/kaggle_tiny'
    if not osp.isdir(run_root):
        raise FileNotFoundError(f'No run folder found at {run_root}')
    run_dirs = sorted(
        [osp.join(run_root, p) for p in os.listdir(run_root)],
        key=lambda p: osp.getmtime(p),
        reverse=True,
    )
    for run_dir in run_dirs:
        best = osp.join(run_dir, 'checkpoints', 'best_mae_bev.pth')
        if osp.isfile(best):
            return best
        chkp_dir = osp.join(run_dir, 'checkpoints')
        if osp.isdir(chkp_dir):
            candidates = sorted(
                [osp.join(chkp_dir, p) for p in os.listdir(chkp_dir) if p.endswith('.pth')],
                key=lambda p: osp.getmtime(p),
                reverse=True,
            )
            if candidates:
                return candidates[0]
    raise FileNotFoundError(f'No checkpoint found under {run_root}')


def to_cuda(data_dict):
    for key in data_dict.keys():
        if key == 'metas':
            continue
        value = data_dict[key]
        if isinstance(value, (dict, edict)):
            data_dict[key] = to_cuda(value)
        elif value is None or value[0] is None:
            continue
        else:
            data_dict[key] = value.cuda(non_blocking=True)
    return data_dict


def world_centers(world_range, shape):
    z, h, w = shape
    x = np.linspace(world_range[0], world_range[3], w, endpoint=False) + (world_range[3] - world_range[0]) / (2 * w)
    y = np.linspace(world_range[1], world_range[4], h, endpoint=False) + (world_range[4] - world_range[1]) / (2 * h)
    zc = np.linspace(world_range[2], world_range[5], z, endpoint=False) + (world_range[5] - world_range[2]) / (2 * z)
    return x, y, zc


def build_figure(pred_density, gt_points, world_range, topk):
    import plotly.graph_objects as go

    pred = pred_density.copy()
    pred[pred < 0] = 0
    flat = pred.reshape(-1)
    nonzero = np.flatnonzero(flat > 0)
    if nonzero.size == 0:
        top_ids = np.array([], dtype=np.int64)
    else:
        top_ids = nonzero[np.argsort(flat[nonzero])[-min(topk, nonzero.size):]]

    z_ids, y_ids, x_ids = np.unravel_index(top_ids, pred.shape)
    xs, ys, zs = world_centers(world_range, pred.shape)
    values = flat[top_ids] if top_ids.size else np.array([])

    fig = go.Figure()
    if top_ids.size:
        sizes = 3 + 12 * (values / (values.max() + 1e-6))
        fig.add_trace(go.Scatter3d(
            x=xs[x_ids], y=ys[y_ids], z=zs[z_ids],
            mode='markers',
            marker=dict(size=sizes, color=values, colorscale='Turbo', opacity=0.55, colorbar=dict(title='Pred density')),
            name='Predicted voxels',
            text=[f'density={v:.4f}' for v in values],
        ))

    if gt_points is not None and len(gt_points) > 0:
        fig.add_trace(go.Scatter3d(
            x=gt_points[:, 0], y=gt_points[:, 1], z=gt_points[:, 2],
            mode='markers',
            marker=dict(size=4, color='black', symbol='diamond', opacity=0.9),
            name='GT head points',
        ))

    fig.update_layout(
        title='CountFormer CityStreet 3D voxel prediction',
        scene=dict(
            xaxis_title='World X',
            yaxis_title='World Y',
            zaxis_title='World Z',
            aspectmode='data',
        ),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def main():
    args = parse_args()
    args.chkp_path = optional_arg(args.chkp_path)
    cfg = Config.fromfile(args.cfg_path)
    checkpoint = args.chkp_path or find_checkpoint()

    dataset_bundle = DATASET.build(cfg.data, datasets=[args.part])
    dataset = dataset_bundle[args.part].dataset
    sample = dataset[args.sample_idx % len(dataset)]
    collate_fn = dataset_bundle[args.part].loader.collate_fn
    batch = edict(collate_fn([sample]))
    batch = to_cuda(batch)

    model = nn.DataParallel(MODELS.build(cfg.model).cuda()).eval()
    state = torch.load(checkpoint, map_location='cpu')
    if 'net' in state:
        HELPERS.get('copy_state_dict')(model.module.state_dict(), state['net'])

    with torch.no_grad(), torch.amp.autocast('cuda', enabled=bool(cfg.model.get('amp', False))):
        pred = model(batch)

    pred_density = pred.vox_density_map[-1][0, 0].detach().float().cpu().numpy()
    gt_points = np.asarray(batch.metas.pt_bev[0], dtype=np.float32)
    fig = build_figure(pred_density, gt_points, cfg.world_range, args.topk)

    os.makedirs(osp.dirname(args.out), exist_ok=True)
    fig.write_html(args.out, include_plotlyjs=True, full_html=True)

    summary = {
        'checkpoint': checkpoint,
        'sample_idx': args.sample_idx,
        'num_gt_points': int(len(gt_points)),
        'pred_count': float(pred_density.clip(min=0).sum() / cfg.scene_density_map_scale),
        'output_html': args.out,
    }
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
