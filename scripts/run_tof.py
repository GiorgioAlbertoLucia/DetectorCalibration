"""
Entry point for TOF mass calibration.

Usage
-----
    python scripts/run_tof.py --config configs/LHC23_pass5_TOF.yaml
"""

import argparse

import sys
sys.path.append('..')
from calibration.common.config import load_config
from calibration.tof.calibrator import TOFCalibrator


def main():
    parser = argparse.ArgumentParser(description='TOF mass calibration')
    parser.add_argument('--config', '-c', required=True,
                        help='Path to YAML config file')
    args = parser.parse_args()

    cfg = load_config(args.config)
    TOFCalibrator(cfg).run()


if __name__ == '__main__':
    main()