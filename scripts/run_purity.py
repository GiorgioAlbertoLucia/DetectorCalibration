"""
Entry point for purity extraction.

Usage
-----
    python scripts/run_purity.py --config configs/LHC23_pass4_purity.yaml
"""

import argparse
from calibration.common.config import load_config
from calibration.purity.purity_fitter import PurityFitter


def main():
    parser = argparse.ArgumentParser(description='Purity extraction')
    parser.add_argument('--config', '-c', required=True,
                        help='Path to YAML config file')
    args = parser.parse_args()

    cfg = load_config(args.config)

    from ROOT import TFile
    from pathlib import Path

    ds_cfg  = cfg['dataset']
    out_cfg = cfg['output']
    out_dir = Path(out_cfg.get('dir', 'output'))
    out_dir.mkdir(parents=True, exist_ok=True)

    label    = ds_cfg['label']
    filename = out_cfg.get('filename', f'purity_{label}.root')
    outpath  = out_dir / filename

    outfile  = TFile(str(outpath), 'RECREATE')
    analysis = PurityFitter(ds_cfg['input_file'], str(outpath))

    pur_cfg   = cfg['purity']
    particles = pur_cfg.get('particles', ['He3', 'Had'])
    detectors = pur_cfg.get('detectors', ['TPC'])

    for particle in particles:
        for detector in detectors:
            det_cfg = pur_cfg.get(particle, {}).get(detector)
            if det_cfg is None:
                print(f'Skipping {particle}/{detector} (no config)')
                continue
            analysis.run(outfile, particle, detector, config=det_cfg)

    outfile.Close()
    print(f'\nResults saved to {outpath}')


if __name__ == '__main__':
    main()