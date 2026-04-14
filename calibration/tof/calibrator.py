"""
TOF cluster-size calibration.

Migrated from TOF.py.  All physics logic is preserved; the public
interface is now the TOFCalibrator class driven by a config dict.

Typical usage (via script)
--------------------------
    from calibration.common.config import load_config
    from calibration.TOF.calibrator import TOFCalibrator

    cfg = load_config('configs/LHC23_pass4_TOF.yaml')
    cal = TOFCalibrator(cfg)
    cal.run()
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm
import pandas as pd
from pathlib import Path
from ROOT import TFile, TF1, TCanvas, TPad, TMath, \
                 RooRealVar, RooAddPdf

from torchic import Dataset, AxisSpec
from torchic.physics import py_BetheBloch, BetheBloch
from torchic.core.graph import create_graph

from calibration.common.config import load_config
from calibration.common.particles import PDG_CODE, PARTICLE_MASS, TREE_SUFFIX
from calibration.common.fit_utils import (
    calibration_fit_slice,
    initialize_means_and_covariances,
)
from calibration.tof.models import init_signal_roofit, init_background_roofit

# ── Default histogram axis bounds (shared across all particles) ───────────────
_X_BOUNDS = {
    'beta_gamma': {'x_min': 0, 'x_max': 5},
    'p':          {'x_min': 0, 'x_max': 6},
}

_X_META = {
    'beta_gamma': {
        'axis_name':  'bg',
        'axis_title': '#beta#gamma',
        'var_name':   'fBetaGamma',
    },
    'p': {
        'axis_name':  'p',
        'axis_title': '#it{p} (GeV/c)',
        'var_name':   'fP',
    },
}


class TOFCalibrator:
    """
    Runs the TOF cluster-size calibration for one dataset.

    Parameters
    ----------
    cfg : dict
        Parsed YAML config (from ``load_config``).
    """

    def __init__(self, cfg: dict, dataset: Dataset):
        self.cfg = cfg
        self.tof_cfg = cfg['tof']
        self.x = self.tof_cfg.get('x_variable', 'beta_gamma')
        self._dataset = dataset
        
        self.signal = RooRealVar('tof_mass', '#it{m}_{TOF}', 0., 5.)
        self.cumulative_pdfs = {}
        self.cumulative_pdf_interp = {}
        self.cumulative_pdfs_binning = None
        
        self.outfile, self.outpath = self._open_outfile()
        print(f'Output → {self.outpath}')

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    @dataset.setter
    def dataset(self, new_dataset: Dataset) -> None:
        self._dataset = new_dataset
    

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, particle) -> None:

        print(f'\n── {particle} ──')

        particle_dir = self.outfile.mkdir(particle)
        self._calibration_routine(particle_dir, particle)
        print(f'\nDone.')

    # ── Dataset helpers ───────────────────────────────────────────────────────
        
    def _open_outfile(self) -> tuple:
        label    = self.cfg['dataset']['label']
        out_dir  = Path(self.cfg['output'].get('dir', 'output'))
        filename = self.cfg['output'].get('filename', f'TOF_{label}.root')
        outpath  = out_dir / filename
        outpath.parent.mkdir(parents=True, exist_ok=True)
        return TFile(str(outpath), 'RECREATE'), outpath

    # ── Core calibration routine ──────────────────────────────────────────────

    def _calibration_routine(
        self, particle_dir, particle: str
    ) -> None:
        """
        Build the 2-D histogram, iterate over x-slices, fit each one,
        then call visualize_fit_results.

        Directly mirrors calibration_routine() in the original TOF.py.
        """
        x_meta  = _X_META[self.x]
        p_cfg   = self.tof_cfg[particle]
        x_bounds = _X_BOUNDS[self.x]

        axis_spec_x = AxisSpec(
            p_cfg['x_nbins'], x_bounds['x_min'], x_bounds['x_max'],
            x_meta['axis_name'], ';;'
        )
        axis_spec_signal = AxisSpec(
            200, 0, 2000, 'cluster_size_cal',
            f';{x_meta["axis_title"]};#it{{m}}_{{T{OF}^{{2}}}}'
        )

        h2_signal = self.dataset.build_th2(
            x_meta['var_name'],
            self.cfg['dataset']['variable_names'][particle]['signal'],
            axis_spec_x, axis_spec_signal,
        )

        signal, signal_pars = init_signal_roofit(
            self.signal, function=self.tof_cfg.get('signal_model', 'gausexp')
        )
        bkg, bkg_pars = init_background_roofit(
            self.signal, function=self.tof_cfg.get('bkg_model', 'exp+exp')
        )

        x_min = p_cfg['x_min_fit']
        x_max = p_cfg['x_max_fit']
        x_bin_min = h2_signal.GetXaxis().FindBin(x_min)
        x_bin_max = h2_signal.GetXaxis().FindBin(x_max)

        fit_results_df = None

        for x_bin in range(x_bin_min, x_bin_max + 1):
            ix          = h2_signal.GetXaxis().GetBinCenter(x_bin)
            x_error     = h2_signal.GetXaxis().GetBinWidth(x_bin) / 2.
            x_low_edge  = h2_signal.GetXaxis().GetBinLowEdge(x_bin)
            x_high_edge = h2_signal.GetXaxis().GetBinLowEdge(x_bin + 1)

            h_signal = h2_signal.ProjectionY(f'signal_{ix:.2f}', x_bin, x_bin, 'e')
            if h_signal.GetEntries() <= 0:
                print(f'  No entries for {particle} at {x_meta["axis_name"]}={ix:.2f}, skipping')
                continue

            model, sig_frac = self._build_model_for_slice(
                signal, signal_pars, bkg, bkg_pars, h_signal, particle, ix, p_cfg
            )

            frame, frame_pull, fit_results = calibration_fit_slice(
                model, h_signal, self.signal, signal_pars, x_low_edge, x_high_edge
            )
            fit_results['x']       = np.abs(ix)
            fit_results['x_error'] = x_error
            
            cumulative_pdf = model.createCdf(self.signal)
            #self.cumulative_pdfs[f'{ix:.1f}'] = cumulative_pdf

            # Sample the CDF over the signal range
            n_points = 1000
            signal_vals = np.linspace(self.signal.getMin(), self.signal.getMax(), n_points)
            prob_vals = np.zeros(n_points)
            for i, val in enumerate(signal_vals):
                self.signal.setVal(val)
                prob_vals[i] = cumulative_pdf.getVal(self.signal)

            self.cumulative_pdf_interp[f'{ix:.1f}'] = interp1d(
                signal_vals, prob_vals, bounds_error=False, fill_value=(1e-10, 1 - 1e-10)
            )

            row = pd.DataFrame.from_dict([fit_results])
            fit_results_df = row if fit_results_df is None \
                             else pd.concat([fit_results_df, row], ignore_index=True)

            self._draw_fit_and_pull(frame, frame_pull, particle_dir, ix)
            del model

        if fit_results_df is None:
            print(f'  No fit results for {particle}, skipping parametrisation.')
            return

        h_pt = h2_signal.ProjectionX(f'pt_{particle}', 1, -1)
        self.cumulative_pdfs_binning = h_pt.Clone(f'cumulative_pdfs_binning_{particle}')
        self._visualize_fit_results(fit_results_df, particle, x_min, x_max, particle_dir)

        particle_dir.cd()
        h2_signal.Write()        


    def _build_model_for_slice(
        self, signal, signal_pars, bkg, bkg_pars, h_signal, particle, ix, p_cfg
    ):
        """Choose signal-only or signal+bkg model and seed initial parameters."""
        x_max_bkg = p_cfg.get('x_max_bkg', 0)
        use_bkg   = (particle == 'Pr' and ix < x_max_bkg) or \
                    (particle == 'He' and ix < x_max_bkg)
        sig_frac = None

        if use_bkg:
            sig_frac = RooRealVar('sig_frac', 'sig_frac', 0.5, 0., 1.)
            model    = RooAddPdf('model', 'signal + bkg', [signal, bkg], [sig_frac])

            if h_signal.GetEntries() > 30:
                est = initialize_means_and_covariances(h_signal, 1, method='kmeans')
                if est is not None:
                    means, variances = est
                    signal_pars['mean'].setVal(means[0])
                    signal_pars['sigma'].setVal(np.sqrt(variances[0]))
                    if 'rlife' in signal_pars:
                        signal_pars['rlife'].setVal(4.)
        else:
            model = signal
            if h_signal.GetEntries() > 30:
                est = initialize_means_and_covariances(h_signal, 1, method='kmeans')
                if est is not None:
                    means, variances = est
                    signal_pars['mean'].setVal(means[0])
                    signal_pars['sigma'].setVal(np.sqrt(variances[0]))
                    if 'rlife' in signal_pars:
                        signal_pars['rlife'].setVal(4.)

        return model, sig_frac

    def _draw_fit_and_pull(self, frame, frame_pull, particle_dir, ix):
        
        canvas = TCanvas(f'cClSizeCosLam_{ix:.2f}', f'cClSizeCosLam_{ix:.2f}', 800, 600)

        upper_pad = TPad(f'pad_upper_{ix:.2f}', f'pad_upper_{ix:.2f}', 0, 0.3 - 0.05, 1, 1)
        upper_pad.Draw()

        upper_pad.cd()
        frame.GetXaxis().SetLabelSize(0)
        frame.GetXaxis().SetTitleSize(0)
        frame.Draw()

        canvas.cd()
        lower_pad = TPad(f'pad_lower_{ix:.2f}', f'pad_lower_{ix:.2f}', 0, 0, 1, 0.3 + 0.024)
        lower_pad.Draw()

        lower_pad.cd()
        frame_pull.SetTitle('')
        frame_pull.GetYaxis().SetTitle('Pull')
        frame_pull.GetYaxis().SetTitleSize(0.06)
        frame_pull.GetYaxis().SetLabelSize(0.06)
        frame_pull.GetXaxis().SetTitleSize(0.06)
        frame_pull.GetXaxis().SetLabelSize(0.06)
        frame_pull.Draw()

        particle_dir.cd()
        canvas.Write()
        
    def _set_bb_params(self, func):
        func.SetParLimits(0, -1000., -1.)    # p1: negative amplitude
        func.SetParLimits(1, 0.1,    5.)     # p2: beta power
        func.SetParLimits(2, 0.5,    5.)     # p3: log argument offset  
        func.SetParLimits(3, 0.1,   20.)     # p4: density correction
        func.SetParLimits(4, 1.5,    3.5)    # p5: exponent

    def _visualize_fit_results(
        self, fit_results_df, particle, x_min, x_max, particle_dir
    ):
        """
        Fit the mean and resolution vs x, derive nσ_TOF, write plots.

        Mirrors visualize_fit_results() in the original TOF.py.
        """
            
        np_bethe_bloch = np.vectorize(py_BetheBloch)
        
        x_meta = _X_META[self.x]

        g_mean = create_graph(
            fit_results_df, 'x', 'mean', 'x_error', 'mean_err',
            'g_mean',
            f';{x_meta["axis_title"]};#LT #it{{m}}_{{TOF}} #GT (GeV/#it{{c}}^{{2}})',
        )
        f_mean = TF1('f_mean', '[0] + [1]*x + [2]*x^2', x_min, x_max, 5)
        f_mean.SetParameters(PARTICLE_MASS[particle], 0.5, 0.1)
        self._set_bb_params(f_mean)
        g_mean.Fit(f_mean, 'RMS+')
        
        g_resolution = create_graph(
            fit_results_df, 'x', 'resolution', 'x_error', 'resolution_err',
            'g_resolution', f';{x_meta["axis_title"]};#sigma / #mu',
        )

        f_resolution = TF1('f_resolution', '[0]', x_min, x_max)
        f_resolution.SetParameters(0.09)
        g_resolution.Fit(f_resolution, 'RMS+')
        
        pid_params = (f_mean.GetParameter(0), f_mean.GetParameter(1), f_mean.GetParameter(2),
                        f_mean.GetParameter(3), f_mean.GetParameter(4), f_resolution.GetParameter(0))
            
        particle_dir.cd()
        g_mean.Write()
        g_resolution.Write()

        self.dataset['fExpTOFSignal'] = np_bethe_bloch(np.abs(self.dataset['fBetaGamma']), *pid_params[:5])
        compute_resolution = lambda betagamma: pid_params[5] #* 1/ (1 + np.exp(-(betagamma-resolution_params[1])/resolution_params[2]))
        compute_resolution_vectorised = np.vectorize(compute_resolution)
        self.dataset['fResolution'] = compute_resolution_vectorised(abs(self.dataset['fBetaGamma']))
        self.dataset['fNSigmaTOF'] = (self.dataset[self.cfg['dataset']['variable_names'][particle]['signal']] - self.dataset['fExpTOFSignal']) / (self.dataset['fExpTOFSignal'] * self.dataset['fResolution'])

        axis_spec_betagamma = AxisSpec(320, -8, 8, 'beta_gamma', ';#beta#gamma;#it{m}_{TOF} (GeV/#it{c})')
        axis_spec_tofsignal = AxisSpec(100, 0, 2000, 'TOF_signal', ';#beta#gamma;#it{m}_{TOF} (GeV/#it{c})')
        axis_spec_nsigmatof = AxisSpec(100, -5, 5, 'nsigma_TOF', ';#beta#gamma;n#sigma_{TOF}')

        h2_nsigmatof = self.dataset.build_th2(f'{x_meta["var_name"]}', 'fNSigmaTOF', axis_spec_betagamma, axis_spec_nsigmatof)
        h2_exptof = self.dataset.build_th2(f'{x_meta["var_name"]}', 'fExpTOFSignal', axis_spec_betagamma, axis_spec_tofsignal)
        h2_tof = self.dataset.build_th2(f'{x_meta["var_name"]}', self.cfg['dataset']['variable_names'][particle]['signal'], 
                                        axis_spec_betagamma, axis_spec_tofsignal)

        f_fit_matter = TF1('f_fit_matter', BetheBloch, x_min, x_max, 5)
        f_fit_matter.SetParameters(*pid_params[:5])
        def BetheBlochAntimatter(x, *params):
            return BetheBloch(-x, *params)
        f_fit_antimatter = TF1('f_fit_antimatter', BetheBlochAntimatter, -x_max, -x_min, 5)
        f_fit_antimatter.SetParameters(*pid_params[:5])

        particle_dir.cd()
        
        h2_tof.Write()
        h2_nsigmatof.Write()
        h2_exptof.Write('exp_tof_signal')
        
        canvas = TCanvas('cNSigmaTOF', 'cNSigmaTOF', 800, 600)
        h2_tof.Draw('colz')
        f_fit_matter.Draw('same')
        f_fit_antimatter.Draw('same')
        canvas.Write()
        
    