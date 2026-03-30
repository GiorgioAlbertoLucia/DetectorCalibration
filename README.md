# DetectorCalibration

Detector calibration (ITS, TPC, TOF) and purity extraction for He³–hadron analyses in ALICE Run 3.

---

## Repository layout

```
DetectorCalibration/
│
├── configs/                    # one YAML per dataset × step
│   ├── LHC23_pass4_ITS.yaml
│   ├── LHC24_pass3_TPC.yaml
│   └── LHC23_pass4_purity.yaml
│
├── calibration/                # installable package
│   ├── common/
│   │   ├── config.py           # YAML loading + validation
│   │   ├── fit_utils.py        # calibration_fit_slice, initialize_means_and_covariances
│   │   └── particles.py        # PDG codes, masses, tree suffixes
│   │
│   ├── its/
│   │   ├── calibrator.py       # ITSCalibrator class
│   │   └── models.py           # RooFit PDF factories
│   │
│   ├── tpc/
│   │   ├── calibrator.py       # TPCCalibrator class
│   │   └── models.py
│   │
│   ├── tof/
│   │   ├── calibrator.py       # TOFCalibrator class
│   │   └── models.py
│   │
│   └── purity/
│       ├── analysis.py         # PurityAnalysis class
│       ├── fitter.py           # PurityFitter class
│       ├── models.py           # SignalModel, BackgroundModel
│       └── sideband_fit.py     # SidebandFitter class
│
├── scripts/                    # thin CLI entry points
│   ├── run_its.py
│   ├── run_tpc.py
│   ├── run_tof.py
│   └── run_purity.py
│
└── output/                     # gitignored; ROOT files land here
```

---

## Installation

```bash
# Clone and install in editable mode (no sys.path hacks needed)
git clone <repo-url>
cd DetectorCalibration
pip install -e .
```

Dependencies: `numpy`, `pandas`, `pyyaml`, `particle`, `torchic`, and a ROOT installation with Python bindings.

---

## Running a calibration

All scripts follow the same pattern:

```bash
python scripts/run_its.py    --config configs/LHC23_pass4_ITS.yaml
python scripts/run_tpc.py    --config configs/LHC24_pass3_TPC.yaml
python scripts/run_tof.py    --config configs/LHC23_pass5_TOF.yaml
python scripts/run_purity.py --config configs/LHC23_pass4_purity.yaml

# TPC special case: read a pre-made TH2 instead of the full dataset
python scripts/run_tpc.py    --config configs/LHC24_pass3_TPC.yaml --use-th2
```

After `pip install -e .` the same commands are available as:

```bash
calib-its    --config configs/LHC23_pass4_ITS.yaml
calib-tpc    --config configs/LHC24_pass3_TPC.yaml
calib-tof    --config configs/LHC23_pass5_TOF.yaml
calib-purity --config configs/LHC23_pass4_purity.yaml
```

---

## Adding a new dataset

1. Copy the closest existing config in `configs/`.
2. Update `dataset.label`, `dataset.input_files`, and any fit ranges that changed.
3. Run.  No code changes required.

---

## Config schema

Every YAML has two mandatory top-level sections:

```yaml
dataset:
  label: <string>          # used in output filenames and plot titles
  input_files: [...]       # list of ROOT file paths (ITS/TPC/TOF)
  input_file: <path>       # single file (purity)
  tree_name: O2he3hadtable
  folder_name: "DF*"

output:
  dir: output/
  # filename: custom_name.root   # optional override
```

Each script then expects its own section (`its:`, `tpc:`, `tof:`, `purity:`). See the example configs for the full set of keys with inline documentation.
