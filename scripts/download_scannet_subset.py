#!/usr/bin/env python
"""
download_scannet_subset.py — fetch only what OccluSynth needs from ScanNet v2.

Staged download strategy:
    --mode frames_25k    : pre-extracted 25k RGB-D frames across all scenes (~6 GB)
                           Use this first; enables adapter training without .sens parsing.
    --mode scenes        : full .sens + meshes for N scenes (~2-3 GB per scene)
                           Use this once your pipeline is verified.
    --mode label_map     : just the label mapping TSV (~1 MB)

Examples:
    # Day 1: get adapter training data
    python download_scannet_subset.py --out_dir ./data/scannet --mode frames_25k

    # Day 5: get 3 full scenes to test the fusion pipeline
    python download_scannet_subset.py --out_dir ./data/scannet --mode scenes --num_scenes 3

    # Day 10: scale up to 20 scenes for proper training
    python download_scannet_subset.py --out_dir ./data/scannet --mode scenes --num_scenes 20

    # Download specific scenes by id (e.g. for the demo video)
    python download_scannet_subset.py --out_dir ./data/scannet --mode scenes \
        --scene_ids scene0011_00 scene0050_00 scene0231_00

By running this you acknowledge the ScanNet Terms of Use.
"""

import argparse
import os
import ssl
import sys
import urllib.request
from typing import List

ssl._create_default_https_context = ssl._create_unverified_context

BASE_URL = 'http://kaldir.vc.cit.tum.de/scannet/'
RELEASE = 'v2/scans'
RELEASE_TASKS = 'v2/tasks'
RELEASE_LIST_URL = BASE_URL + 'v2/scans.txt'                # all v2 train+val scene ids
LABEL_MAP_FILE = 'scannetv2-labels.combined.tsv'
FRAMES_25K_FILE = 'scannet_frames_25k.zip'                  # ~5.6 GB, pre-extracted RGB-D

# File types OccluSynth actually consumes.
ESSENTIAL_FILETYPES = [
    '.sens',                   # RGB + depth + intrinsics + poses (THE big one)
    '.txt',                    # scene metadata (intrinsics summary, axis-aligned bbox)
    '_vh_clean_2.ply',         # GT mesh — completion target
    '_vh_clean_2.labels.ply',  # GT mesh with NYU40 semantic labels per vertex
]

# Optional extras — only useful if you decide to do instance-level work.
OPTIONAL_FILETYPES = [
    '_vh_clean.aggregation.json',
    '_vh_clean_2.0.010000.segs.json',
]


def _progress(count, block_size, total_size, label):
    if total_size <= 0:
        return
    pct = min(100.0, count * block_size * 100.0 / total_size)
    mb = count * block_size / (1024 * 1024)
    total_mb = total_size / (1024 * 1024)
    sys.stdout.write(f'\r    {label}: {mb:7.1f} / {total_mb:7.1f} MB  ({pct:5.1f}%)')
    sys.stdout.flush()


def download_file(url: str, out_file: str) -> None:
    """Download with progress + resume-safe via .tmp rename. Skips if file already exists."""
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    if os.path.isfile(out_file):
        size_mb = os.path.getsize(out_file) / (1024 * 1024)
        print(f'  [skip] {os.path.basename(out_file)} ({size_mb:.1f} MB) — already on disk')
        return

    tmp_file = out_file + '.tmp'
    label = os.path.basename(out_file)
    try:
        urllib.request.urlretrieve(
            url, tmp_file,
            reporthook=lambda c, b, t: _progress(c, b, t, label),
        )
        sys.stdout.write('\n')
        os.rename(tmp_file, out_file)
    except Exception as e:
        if os.path.isfile(tmp_file):
            os.remove(tmp_file)
        sys.stdout.write(f'\n  [FAIL] {url}\n         {e}\n')
        raise


def fetch_scene_list() -> List[str]:
    print(f'Fetching full v2 scene list from {RELEASE_LIST_URL}')
    with urllib.request.urlopen(RELEASE_LIST_URL) as f:
        scenes = [line.decode('utf-8').strip() for line in f if line.strip()]
    print(f'  -> {len(scenes)} scenes available')
    return scenes


def download_scene(scan_id: str, scans_dir: str, file_types: List[str]) -> None:
    print(f'\n[{scan_id}]')
    scene_dir = os.path.join(scans_dir, scan_id)
    for ft in file_types:
        url = f'{BASE_URL}{RELEASE}/{scan_id}/{scan_id}{ft}'
        out_file = os.path.join(scene_dir, f'{scan_id}{ft}')
        try:
            download_file(url, out_file)
        except Exception:
            print(f'  -> continuing to next file')


def download_label_map(out_dir: str) -> None:
    print('\n[label map]')
    url = f'{BASE_URL}{RELEASE_TASKS}/{LABEL_MAP_FILE}'
    out_file = os.path.join(out_dir, LABEL_MAP_FILE)
    download_file(url, out_file)


def download_frames_25k(out_dir: str) -> None:
    print('\n[scannet_frames_25k.zip — ~5.6 GB]')
    url = f'{BASE_URL}{RELEASE_TASKS}/{FRAMES_25K_FILE}'
    out_file = os.path.join(out_dir, 'tasks', FRAMES_25K_FILE)
    download_file(url, out_file)
    print(f'\nDownloaded. Unzip with:  unzip {out_file} -d {os.path.join(out_dir, "tasks")}')


def confirm(prompt: str) -> None:
    try:
        input(prompt + ' (Enter to continue, Ctrl-C to abort) ')
    except KeyboardInterrupt:
        print('\nAborted.')
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--out_dir', required=True, help='root data directory')
    p.add_argument('--mode', required=True,
                   choices=['frames_25k', 'scenes', 'label_map'],
                   help='what to download')
    p.add_argument('--num_scenes', type=int, default=10,
                   help='how many scenes to grab in --mode scenes (default 10)')
    p.add_argument('--scene_ids', nargs='+',
                   help='specific scene ids (overrides --num_scenes)')
    p.add_argument('--include_optional', action='store_true',
                   help='also fetch segmentation/aggregation JSONs')
    p.add_argument('--start', type=int, default=0,
                   help='start index into the scene list (use to fetch a different slice next time)')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Always grab the label map — it's tiny and the dataloader needs it.
    download_label_map(args.out_dir)

    if args.mode == 'label_map':
        return

    if args.mode == 'frames_25k':
        print('\nThis downloads ~5.6 GB of pre-extracted RGB-D frames.')
        print('Recommended for VGGT adapter training (Days 3-5).')
        confirm('OK?')
        download_frames_25k(args.out_dir)
        print('\nDone.')
        return

    # mode == 'scenes'
    file_types = list(ESSENTIAL_FILETYPES)
    if args.include_optional:
        file_types += OPTIONAL_FILETYPES

    if args.scene_ids:
        scenes = args.scene_ids
    else:
        all_scenes = fetch_scene_list()
        scenes = all_scenes[args.start:args.start + args.num_scenes]

    est_gb = len(scenes) * 2.5  # rough — .sens dominates at ~1-4 GB per scene
    print(f'\nWill fetch {len(file_types)} file types for {len(scenes)} scene(s).')
    print(f'Estimated size: ~{est_gb:.0f} GB  (mostly .sens files)')
    print(f'First scenes: {scenes[:5]}{"..." if len(scenes) > 5 else ""}')
    confirm('Proceed?')

    scans_dir = os.path.join(args.out_dir, 'scans')
    for sid in scenes:
        download_scene(sid, scans_dir, file_types)

    print(f'\nDone. Data in {args.out_dir}/scans/')
    print('Next step: extract .sens files using ScanNet\'s SensReader')
    print('  https://github.com/ScanNet/ScanNet/tree/master/SensReader/python')


if __name__ == '__main__':
    main()