"""
ITS cluster-size calibration.

Migrated from ITS.py.  All physics logic is preserved; the public
interface is now the ITSCalibrator class driven by a config dict.

Typical usage (via script)
--------------------------
    from calibration.common.config import load_config
    from calibration.its.calibrator import ITSCalibrator

    cfg = load_config('configs/LHC23_pass4_ITS.yaml')
    cal = ITSCalibrator(cfg)
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
from torchic.core.graph import create_graph
from torchic.physics.ITS import average_cluster_size, expected_cluster_size

from calibration.common.config import load_config
from calibration.common.particles import PDG_CODE, PARTICLE_MASS, TREE_SUFFIX
from calibration.common.fit_utils import (
    calibration_fit_slice,
    initialize_means_and_covariances,
)
from calibration.its.models import init_signal_roofit, init_background_roofit

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


class ITSCalibrator:
    """
    Runs the ITS cluster-size calibration for one dataset.

    Parameters
    ----------
    cfg : dict
        Parsed YAML config (from ``load_config``).
    """

    def __init__(self, cfg: dict, dataset: Dataset):
        self.cfg = cfg
        self.its_cfg = cfg['its']
        self.x = self.its_cfg.get('x_variable', 'beta_gamma')
        self._dataset = dataset
        
        self.clsize = RooRealVar('clsize', '#LT Cluster size #GT #LT cos#lambda #GT', 0., 15.)
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
        filename = self.cfg['output'].get('filename', f'ITS_{label}.root')
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

        Directly mirrors calibration_routine() in the original ITS.py.
        """
        x_meta  = _X_META[self.x]
        p_cfg   = self.its_cfg[particle]
        x_bounds = _X_BOUNDS[self.x]

        axis_spec_x = AxisSpec(
            p_cfg['x_nbins'], x_bounds['x_min'], x_bounds['x_max'],
            x_meta['axis_name'], ';;'
        )
        axis_spec_clsize = AxisSpec(
            75, 0, 15, 'cluster_size_cal',
            f';{x_meta["axis_title"]};#LT ITS Cluster Size #GT #times cos #LT #lambda #GT'
        )

        h2_clsize = self.dataset.build_th2(
            x_meta['var_name'],
            self.cfg['dataset']['variable_names'][particle]['avg_cl_size_cosl'],
            axis_spec_x, axis_spec_clsize,
        )

        signal, signal_pars = init_signal_roofit(
            self.clsize, function=self.its_cfg.get('signal_model', 'gausexp')
        )
        bkg, bkg_pars = init_background_roofit(
            self.clsize, particle, function=self.its_cfg.get('bkg_model', 'gausexp')
        )

        x_min = p_cfg['x_min_fit']
        x_max = p_cfg['x_max_fit']
        x_bin_min = h2_clsize.GetXaxis().FindBin(x_min)
        x_bin_max = h2_clsize.GetXaxis().FindBin(x_max)

        fit_results_df = None

        for x_bin in range(x_bin_min, x_bin_max + 1):
            ix          = h2_clsize.GetXaxis().GetBinCenter(x_bin)
            x_error     = h2_clsize.GetXaxis().GetBinWidth(x_bin) / 2.
            x_low_edge  = h2_clsize.GetXaxis().GetBinLowEdge(x_bin)
            x_high_edge = h2_clsize.GetXaxis().GetBinLowEdge(x_bin + 1)

            h_clsize = h2_clsize.ProjectionY(f'clsize_{ix:.2f}', x_bin, x_bin, 'e')
            if h_clsize.GetEntries() <= 0:
                print(f'  No entries for {particle} at {x_meta["axis_name"]}={ix:.2f}, skipping')
                continue

            model, sig_frac = self._build_model_for_slice(
                signal, signal_pars, bkg, bkg_pars, h_clsize, particle, ix, p_cfg
            )

            frame, frame_pull, fit_results = calibration_fit_slice(
                model, h_clsize, self.clsize, signal_pars, x_low_edge, x_high_edge
            )
            fit_results['x']       = np.abs(ix)
            fit_results['x_error'] = x_error
            
            cumulative_pdf = model.createCdf(self.clsize)
            #self.cumulative_pdfs[f'{ix:.1f}'] = cumulative_pdf

            # Sample the CDF over the clsize range
            n_points = 1000
            clsize_vals = np.linspace(self.clsize.getMin(), self.clsize.getMax(), n_points)
            prob_vals = np.zeros(n_points)
            for i, val in enumerate(clsize_vals):
                self.clsize.setVal(val)
                prob_vals[i] = cumulative_pdf.getVal(self.clsize)

            self.cumulative_pdf_interp[f'{ix:.1f}'] = interp1d(
                clsize_vals, prob_vals, bounds_error=False, fill_value=(1e-10, 1 - 1e-10)
            )

            row = pd.DataFrame.from_dict([fit_results])
            fit_results_df = row if fit_results_df is None \
                             else pd.concat([fit_results_df, row], ignore_index=True)

            self._draw_fit_and_pull(frame, frame_pull, particle_dir, ix)
            del model

        if fit_results_df is None:
            print(f'  No fit results for {particle}, skipping parametrisation.')
            return

        h_pt = h2_clsize.ProjectionX(f'pt_{particle}', 1, -1)
        self.cumulative_pdfs_binning = h_pt.Clone(f'cumulative_pdfs_binning_{particle}')
        self._visualize_fit_results(fit_results_df, particle, x_min, x_max, particle_dir)

        particle_dir.cd()
        h2_clsize.Write()        


    def _build_model_for_slice(
        self, signal, signal_pars, bkg, bkg_pars, h_clsize, particle, ix, p_cfg
    ):
        """Choose signal-only or signal+bkg model and seed initial parameters."""
        x_max_bkg = p_cfg.get('x_max_bkg', 0)
        use_bkg   = (particle == 'Pr' and ix < x_max_bkg) or \
                    (particle == 'He' and ix < x_max_bkg)
        sig_frac = None

        if use_bkg:
            sig_frac = RooRealVar('sig_frac', 'sig_frac', 0.5, 0., 1.)
            model    = RooAddPdf('model', 'signal + bkg', [signal, bkg], [sig_frac])

            if h_clsize.GetEntries() > 30:
                est = initialize_means_and_covariances(h_clsize, 2, method='kmeans')
                if est is not None:
                    means, variances = est
                    signal_pars['mean'].setVal(means[1])
                    signal_pars['sigma'].setVal(np.sqrt(variances[1]))
                    #signal_pars['mean'].setRange(means[1] - 3*np.sqrt(variances[1]), means[1] + 3*np.sqrt(variances[1]))
                    bkg_pars['mean'].setVal(means[0])
                    bkg_pars['sigma'].setVal(np.sqrt(variances[0]))
                    #bkg_pars['mean'].setRange(means[0] - 3*np.sqrt(variances[0]), means[0] + 3*np.sqrt(variances[0]))
        else:
            model = signal
            if h_clsize.GetEntries() > 30:
                est = initialize_means_and_covariances(h_clsize, 1, method='kmeans')
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

    def _visualize_fit_results(
        self, fit_results_df, particle, x_min, x_max, particle_dir
    ):
        """
        Fit the mean and resolution vs x, derive nσ_ITS, write plots.

        Mirrors visualize_fit_results() in the original ITS.py.
        """

        x_meta = _X_META[self.x]

        g_mean = create_graph(
            fit_results_df, 'x', 'mean', 'x_error', 'mean_err',
            'g_mean',
            f';{x_meta["axis_title"]};#LT ITS Cluster Size #GT #times cos #LT #lambda #GT',
        )
        f_mean = TF1('simil_bethe_bloch_func', '[0]/x^[1] + [2]', x_min, x_max)
        f_mean.SetParameters(*(2.3, 1.7, 4.5) if particle == 'He' else (2.6, 2., 2))
        f_mean.SetParLimits(0, 0, 4)
        g_mean.Fit(f_mean, 'RMS+')

        fit_results_df['log_mean']     = np.log(fit_results_df['mean'])
        fit_results_df['log_mean_err'] = fit_results_df['mean_err'] / fit_results_df['mean']
        g_log_mean = create_graph(
            fit_results_df, 'x', 'log_mean', 'x_error', 'log_mean_err',
            'g_log_mean',
            f';{x_meta["axis_title"]};ln(#LT ITS Cluster Size #GT #times cos #LT #lambda #GT)',
        )
        f_mean_log = TF1('log_bethe_bloch_func', 'log([0]/x^[1] + [2])', x_min, x_max)
        f_mean_log.SetParameters(*(2.3, 1.7, 4.5) if particle == 'He' else (2.6, 2., 2))
        g_log_mean.Fit(f_mean_log, 'RMS+')

        g_sigma = create_graph(
            fit_results_df, 'x', 'sigma', 'x_error', 'sigma_err',
            'g_sigma', f';{x_meta["axis_title"]};#sigma',
        )
        g_resolution = create_graph(
            fit_results_df, 'x', 'resolution', 'x_error', 'resolution_err',
            'g_resolution', f';{x_meta["axis_title"]};#sigma / #mu',
        )

        if particle == 'He':
            f_resolution = TF1('resolution_fit', '[0] + x*[1] + x*x*[2]', x_min, x_max)
            f_resolution.SetParameters(0.116, 0., 0)
            f_resolution.SetParLimits(0, 0, 1)
            f_resolution.SetParLimits(1, -0.01, 0.01)
            f_resolution.FixParameter(2, 0)
        else:
            f_resolution = TF1('resolution_fit', '[0]*TMath::Erf((x - [1])/[2])', x_min, x_max)
            f_resolution.SetParameters(0.155, 0.5, 0.1)
        g_resolution.Fit(f_resolution, 'RMS+')

        pid_params = (
            f_mean.GetParameter(0), f_mean.GetParameter(1), f_mean.GetParameter(2),
            f_resolution.GetParameter(0), f_resolution.GetParameter(1), f_resolution.GetParameter(2),
        )

        self.dataset[f'fExpClSizeCosLam'] = expected_cluster_size(
            self.dataset[f'fBetaGamma'], pid_params
        )
        if particle == 'He':
            self.dataset[f'fSigmaITS'] = (
                self.dataset[f'fExpClSizeCosLam']
                * (pid_params[3] + pid_params[4] * self.dataset[f'fBetaGamma'])
            )
        else:
            self.dataset[f'fSigmaITS'] = self.dataset[f'fExpClSizeCosLam'] * pid_params[3] * norm.cdf(
                (self.dataset[f'fBetaGamma'] - pid_params[4]) / pid_params[5])

        self.dataset[f'fNSigmaITS'] = (
            (self.dataset[f'fAvgClSizeCosLam'] - self.dataset[f'fExpClSizeCosLam'])
            / self.dataset[f'fSigmaITS']
        )

        #self.dataset[f'fNSigmaITS'] = -999.
        #nsigmas = []
        #for betagamma, avg_cl_size_coslam, exp_cl_size_coslam, sigma in zip(
        #    self.dataset['fBetaGamma'], self.dataset['fAvgClSizeCosLam'], 
        #    self.dataset[f'fExpClSizeCosLam'], self.dataset[f'fSigmaITS']):
        #    nsigma = self._get_nsigma(betagamma, avg_cl_size_coslam, exp_cl_size_coslam, sigma)
        #    nsigmas.append(nsigma)
        #self.dataset[f'fNSigmaITS'] = nsigmas
        
        #nsigmas = np.full(len(self.dataset), -999.)
        #chunk_size = 1000000
        #for start in range(0, len(self.dataset), chunk_size):
        #    end = min(start + chunk_size, len(self.dataset))
        #    nsigmas[start:end] = self._get_nsigma_vectorized(
        #        self.dataset['fBetaGamma'].iloc[start:end],
        #        self.dataset['fAvgClSizeCosLam'].iloc[start:end],
        #        self.dataset['fExpClSizeCosLam'].iloc[start:end],
        #        self.dataset['fSigmaITS'].iloc[start:end]
        #    )
        #self.dataset['fNSigmaITS'] = nsigmas
            
        print(f'  Derived nσ_ITS for {len(self.dataset)} entries.')
        
        axis_spec_bg     = AxisSpec(50, 0, 5, x_meta['axis_name'], f'{x_meta["axis_title"]}')
        axis_spec_nsigma = AxisSpec(
            100, -5, 5, 'nsigma', f'n#sigma_{{ITS}}'
        )
        h2_nsigma = self.dataset.build_th2(
            f'fBetaGamma', f'fNSigmaITS',
            axis_spec_bg, axis_spec_nsigma,
        )
        print(f'  Built nσ_ITS histogram with {h2_nsigma.GetEntries()} entries.')

        c_mean      = TCanvas('c_mean',      'c_mean',      800, 600)
        c_log_mean  = TCanvas('c_log_mean',  'c_log_mean',  800, 600)
        c_resolution = TCanvas('c_resolution', 'c_resolution', 800, 600)

        c_mean.cd();      g_mean.Draw('ap');      f_mean.Draw('same')
        c_log_mean.cd();  g_log_mean.Draw('ap');  f_mean_log.Draw('same')
        c_resolution.cd(); g_resolution.Draw('ap'); f_resolution.Draw('same')

        particle_dir.cd()
        c_mean.Write()
        c_log_mean.Write()
        g_sigma.Write()
        c_resolution.Write()
        h2_nsigma.Write()
        
    def _get_nsigma(self, x, cluster_size_coslam, expected_cluster_size, sigma):
        
        x_bin_center = self.cumulative_pdfs_binning.GetBinCenter(self.cumulative_pdfs_binning.FindBin(x))
        interp = self.cumulative_pdf_interp.get(f'{x_bin_center:.1f}')

        if interp is None:
            return (cluster_size_coslam - expected_cluster_size) / sigma if sigma > 0 else -999

        prob = float(interp(cluster_size_coslam))
        prob = max(1e-10, min(1 - 1e-10, prob))
        return norm.ppf(prob)
    
    def _get_nsigma_vectorized(self, x, cluster_size_coslam, expected_cluster_size=None, sigma=None):
        return np.vectorize(self._get_nsigma)(x, cluster_size_coslam, expected_cluster_size, sigma)
        