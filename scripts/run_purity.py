"""
Entry point for purity extraction.

Usage
-----
    python scripts/run_purity.py --config configs/LHC23_pass4_purity.yaml
"""

import argparse
from ROOT import TFile

import sys
sys.path.append('..')
from calibration.common.config import load_config
from calibration.purity.purity_analysis import PurityAnalysis


def main():
    parser = argparse.ArgumentParser(description='Purity extraction')
    parser.add_argument('--config', '-c', required=True,
                        help='Path to YAML config file')
    args = parser.parse_args()

    cfg = load_config(args.config)


    input_file_path  = cfg['dataset']['input_files'][0]
    output_file_path = cfg['output']['dir'] + f"/{cfg['dataset']['label']}_purity.root"
    
    outfile  = TFile(str(output_file_path), 'RECREATE')

    cfg_purity = cfg['purity']
    particles = cfg_purity['particles']

    for particle in particles:
        
        detectors = cfg_purity[particle]['detectors']
        for detector in detectors:
            
            analysis = PurityAnalysis(input_file_path, output_file_path)
            analysis.run(outfile, particle, detector, cfg_purity[particle][detector])
        
    outfile.Close()
    print(f'\nResults saved to {output_file_path}')


if __name__ == '__main__':
    main()