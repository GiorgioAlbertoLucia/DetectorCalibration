"""
Entry point for ITS cluster-size calibration.

Usage
-----
    python scripts/run_its.py --config configs/LHC23_pass4_ITS.yaml
    # or, after `pip install -e .`:
    calib-its --config configs/LHC23_pass4_ITS.yaml
"""

import argparse
import numpy as np
from ROOT import gStyle
from torchic.physics.ITS import average_cluster_size
from torchic import Dataset

import sys
sys.path.append('..')
from calibration.common.config import load_config
from calibration.common.particles import PDG_CODE, PARTICLE_MASS, TREE_SUFFIX
from calibration.its.calibrator import ITSCalibrator

def load_dataset(cfg) -> Dataset:
    ds_cfg = cfg['dataset']
    return Dataset.from_root(
        ds_cfg['input_files'],
        tree_name=ds_cfg['tree_name'],
        folder_name=ds_cfg['folder_name'],
    )

def prepare_dataset(dataset, particle:str, cfg):

    mode = cfg['its']['cluster_size_mode']
    cfg = cfg['dataset']['variable_names'][particle]
    if particle == 'He': 
        dataset.query(f'{cfg["pid_for_tracking"]} == 7', inplace=True)
    
    dataset['fP'] = dataset[cfg['pt']] * np.cosh(dataset[cfg['eta']])

    dataset[cfg['cluster_size']] = np.array(
        dataset[cfg['cluster_size']], np.uint64
    )

    do_truncated = (mode == 'truncated')
    dataset[f'fAvgClusterSize'], dataset[f'fNHitsIts'] = \
        average_cluster_size(dataset[cfg['cluster_size']], do_truncated=do_truncated)

    dataset.query(f'fNHitsIts > 5', inplace=True)
    dataset[f'fAvgClSizeCosLam'] = (
        dataset[f'fAvgClusterSize'] / np.cosh(dataset[cfg['eta']])
    )
    mass = PARTICLE_MASS[particle]
    dataset[f'fBetaGamma'] = np.abs(dataset['fP']) / mass
    dataset['fP']         = np.abs(dataset['fP'])

    cfg['p'] = 'fP'
    cfg['betagamma'] = f'fBetaGamma'
    cfg['avg_cl_size_cosl'] = f'fAvgClSizeCosLam'
    cfg['n_hits_its'] = f'fNHitsIts'

    return dataset

def main():
    parser = argparse.ArgumentParser(description='ITS cluster-size calibration')
    parser.add_argument(
        '--config', '-c', required=True,
        help='Path to YAML config file (e.g. configs/LHC23_pass4_ITS.yaml)',
    )
    args = parser.parse_args()

    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)

    cfg = load_config(args.config)

    dataset = load_dataset(cfg)
    calibrator = ITSCalibrator(cfg, dataset)

    for particle in cfg['its']['particles']:

        prepare_dataset(dataset, particle, cfg)
        calibrator.dataset = dataset
        calibrator.run(particle)



if __name__ == '__main__':
    main()