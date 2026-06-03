"""
Entry point for TPC Bethe-Bloch calibration.

Usage
-----
    python scripts/run_tpc.py --config configs/LHC24_pass3_TPC.yaml
    python scripts/run_tpc.py --config configs/LHC24_pass3_TPC.yaml --use-th2
"""

import argparse
import numpy as np
from ROOT import gStyle
from torchic.physics.ITS import average_cluster_size
from torchic import Dataset
from torchic.physics.particles import PARTICLES

import sys
sys.path.append('..')
from calibration.common.config import load_config
from calibration.tpc.calibrator import TPCCalibrator

def load_dataset(cfg) -> Dataset:
    ds_cfg = cfg['dataset']
    return Dataset.from_root(
        ds_cfg['input_files'],
        tree_name=ds_cfg['tree_name'],
        folder_name=ds_cfg['folder_name'],
    )

def prepare_dataset(dataset, particle:str, cfg):

    print(f'{dataset.columns=}')
    print(f'{dataset["fSignalTPCHe3"]=}')
    is_mc = cfg['dataset'].get('is_mc', False)
    cfg = cfg['dataset']['variable_names'][particle]
    
    if particle == 'He': 
        dataset.query(f'{cfg["pid_for_tracking"]} == 7', inplace=True)
    
    if 'p' not in cfg.keys():
        dataset['fP'] = dataset[cfg['pt']] * np.cosh(dataset[cfg['eta']])
        cfg['p'] = 'fP'
    else:
        dataset['fP'] = dataset[cfg['p']]
        if particle == 'He':
            dataset['fP'] = dataset['fP'] * 2 

    dataset[cfg['cluster_size']] = np.array(
        dataset[cfg['cluster_size']], np.uint64
    )

    dataset[f'fAvgClusterSize'], dataset[f'fNHitsIts'] = \
        average_cluster_size(dataset[cfg['cluster_size']], do_truncated=True)

    dataset.query(f'fNHitsIts > 5', inplace=True)
    if particle == 'He':
        dataset.query('fAvgClusterSize > 5', inplace=True)
    
    mass = PARTICLES[particle].mass
    dataset[f'fBetaGamma'] = np.abs(dataset['fP']) / mass
    dataset['fP']          = np.abs(dataset['fP'])

    cfg['betagamma'] = f'fBetaGamma'
    cfg['avg_cl_size_cosl'] = f'fAvgClSizeCosLam'
    cfg['n_hits_its'] = f'fNHitsIts'

    return dataset

def main():
    parser = argparse.ArgumentParser(description='TPC calibration')
    parser.add_argument(
        '--config', '-c', required=True,
        help='Path to YAML config file (e.g. configs/LHC23_pass4_TPC.yaml)',
    )
    args = parser.parse_args()

    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)

    cfg = load_config(args.config)

    dataset = load_dataset(cfg)
    calibrator = TPCCalibrator(cfg, dataset)

    for particle in cfg['tpc']['particles']:

        prepare_dataset(dataset, particle, cfg)
        calibrator.dataset = dataset
        calibrator.run(particle)


if __name__ == '__main__':
    main()