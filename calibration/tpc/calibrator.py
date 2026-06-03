"""
TPC cluster-size calibration.

Migrated from TPC.py.  All physics logic is preserved; the public
interface is now the TPCCalibrator class driven by a config dict.

Typical usage (via script)
--------------------------
    from calibration.common.config import load_config
    from calibration.TPC.calibrator import TPCCalibrator

    cfg = load_config('configs/LHC23_pass4_TPC.yaml')
    cal = TPCCalibrator(cfg)
    cal.run()
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm
import pandas as pd
from pathlib import Path
from ROOT import TFile, TF1, TCanvas, TPad, TMath, TGraphErrors, \
                 RooRealVar, RooAddPdf

from torchic import Dataset, AxisSpec
from torchic.physics import py_BetheBloch, BetheBloch
from torchic.core.graph import create_graph
from torchic.utils.root import set_root_object

from calibration.common.fit_utils import (
    calibration_fit_slice,
    initialize_means_and_covariances,
)
from calibration.tpc.models import init_signal_roofit, init_background_roofit

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


class TPCCalibrator:
    """
    Runs the TPC cluster-size calibration for one dataset.

    Parameters
    ----------
    cfg : dict
        Parsed YAML config (from ``load_config``).
    """

    def __init__(self, cfg: dict, dataset: Dataset):
        self.cfg = cfg
        self.tpc_cfg = cfg['tpc']
        self.x = self.tpc_cfg.get('x_variable', 'beta_gamma')
        self._dataset = dataset
        
        self.signal = RooRealVar('tpc_signal', 'd#it{E}/d#it{x}', 0., 2000.)
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
        filename = self.cfg['output'].get('filename', f'TPC_{label}.root')
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

        Directly mirrors calibration_routine() in the original TPC.py.
        """
        x_meta  = _X_META[self.x]
        p_cfg   = self.tpc_cfg[particle]
        x_bounds = _X_BOUNDS[self.x]

        axis_spec_x = AxisSpec(
            p_cfg['x_nbins'], x_bounds['x_min'], x_bounds['x_max'],
            x_meta['axis_name'], ';;'
        )
        axis_spec_signal = AxisSpec(
            200, 0, 2000, 'cluster_size_cal',
            f';{x_meta["axis_title"]};d#it{{E}}/d#it{{x}}'
        )

        h2_signal = self.dataset.build_th2(
            x_meta['var_name'],
            self.cfg['dataset']['variable_names'][particle]['signal'],
            axis_spec_x, axis_spec_signal,
        )

        signal, signal_pars = init_signal_roofit(
            self.signal, function=self.tpc_cfg.get('signal_model', 'gausexp')
        )
        bkg, bkg_pars = init_background_roofit(
            self.signal, function=self.tpc_cfg.get('bkg_model', 'gausexp')
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
                est = initialize_means_and_covariances(h_signal, 2, method='kmeans')
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
        func.SetParLimits(3, 0.5,    2.)     # p4: density correction
        func.SetParLimits(4, 1.5,    3.5)    # p5: exponent

    def _visualize_fit_results(
        self, fit_results_df, particle, x_min, x_max, particle_dir
    ):
        """
        Fit the mean and resolution vs x, derive nσ_TPC, write plots.

        Mirrors visualize_fit_results() in the original TPC.py.
        """
            
        np_bethe_bloch = np.vectorize(py_BetheBloch)
        
        x_meta = _X_META[self.x]

        g_mean = create_graph(
            fit_results_df, 'x', 'mean', 'x_error', 'mean_err',
            'g_mean',
            f';{x_meta["axis_title"]};#LT d#it{{E}}/d#it{{x}} #GT (a.u.)',
        )
        f_mean = TF1('f_mean', BetheBloch, x_min, x_max, 5)
        f_mean.SetParameters(-241.4902, 0.374245, 1.397847, 1.0782504, 2.048336)
        self._set_bb_params(f_mean)
        g_mean.Fit(f_mean, 'RMS+')
        chi2_mean = f_mean.GetChisquare() / f_mean.GetNDF()
        
        fit_results_df['log_mean'] = np.log(fit_results_df['mean'])
        fit_results_df['log_mean_err'] = fit_results_df['mean_err'] / fit_results_df['mean']
        g_log_mean = create_graph(
            fit_results_df, 'x', 'log_mean', 'x_error', 'log_mean_err',
            'g_log_mean', f';{x_meta["axis_title"]};ln(#LT d#it{{E}}/d#it{{x}} #GT)',
        ) 
        
        f_log_mean_lowbg = TF1('f_mean_lowbg', lambda x, *params: np.log(max(BetheBloch(x, *params), 1e-10)),
                               x_min, 1.5, 5)
        f_log_mean_lowbg.SetParameters(*[f_mean.GetParameter(i) for i in range(5)])
        self._set_bb_params(f_log_mean_lowbg)
        g_mean.Fit(f_log_mean_lowbg, 'RMS+')
        chi2_log_mean_lowbg = f_log_mean_lowbg.GetChisquare() / f_log_mean_lowbg.GetNDF()

        f_log_mean = TF1('f_log_mean', lambda x, *params: np.log(max(BetheBloch(x, *params), 1e-10)),
                         x_min, x_max, 5)
        f_log_mean.SetParameters(*[f_log_mean_lowbg.GetParameter(i) for i in range(5)] if \
            chi2_log_mean_lowbg < chi2_mean else [f_mean.GetParameter(i) for i in range(5)])
        self._set_bb_params(f_log_mean)
        g_log_mean.Fit(f_log_mean, 'RMS+')
        
        g_residuals = TGraphErrors(fit_results_df.shape[0])
        set_root_object(g_residuals, name='g_mean_residuals', title=f';{x_meta["axis_title"]};Residuals (a.u.)')
        for i, row in fit_results_df.iterrows():
            x = row['x']
            y = row['mean']
            y_fit = np.exp(f_log_mean.Eval(x))
            residual = y - y_fit
            g_residuals.SetPoint(i, x, residual)
            g_residuals.SetPointError(i, row['x_error'], row['mean_err'])
        f_residuals = TF1('f_residuals', 
            '[0]*exp(-0.5*((x-[1])/[2])**2) + [3]*exp(-0.5*((x-[4])/[5])**2)',
            x_min, 1.5)
            #x_min, x_max)
        residuals_cfg   = self.tpc_cfg[particle]['residuals']
        
        f_residuals.SetParameters(residuals_cfg.get('par0', [38])[0], 
                                  residuals_cfg.get('par1', [0.65])[0], 
                                  residuals_cfg.get('par2', [0.05])[0],   # positive spike
                                  residuals_cfg.get('par3', [-10])[0], 
                                  residuals_cfg.get('par4', [0.9])[0],  
                                  residuals_cfg.get('par5', [0.1])[0])    # negative dip
        f_residuals.SetParLimits(0, *(residuals_cfg.get('par0',(0, 0, 50))[1:] ))       # positive spike amplitude
        f_residuals.SetParLimits(1, *(residuals_cfg.get('par1',(1, 0.1, 0.8))[1:] ))         # positive spike position
        f_residuals.SetParLimits(2, *(residuals_cfg.get('par2',(2, 0., 0.1))[1:] ))       # positive spike position
        f_residuals.SetParLimits(3, *(residuals_cfg.get('par3',(3, -30, 0))[1:] ))       # positive spike position
        f_residuals.SetParLimits(4, *(residuals_cfg.get('par4',(4, 0.8, 1.2))[1:] ))        # negative dip position
        g_residuals.Fit(f_residuals, 'RMS+')
        residual_params = [f_residuals.GetParameter(i) for i in range(6)]
        
        g_resolution = create_graph(
            fit_results_df, 'x', 'resolution', 'x_error', 'resolution_err',
            'g_resolution', f';{x_meta["axis_title"]};#sigma / #mu',
        )

        f_resolution = TF1('f_resolution', '[0]', x_min, x_max)
        f_resolution.SetParameters(0.09)
        
        #f_resolution = TF1('f_resolution', '[0] + [1] / (1 + TMath::Exp(- (x - [2]) / [3] ))', x_min, x_max)
        #f_resolution.SetParameters(0.04, 0.02, 0.5, 0.1)
        #f_resolution.SetParLimits(0, 0.01, 0.2)   # baseline resolution
        #f_resolution.SetParLimits(1, 0.001, 0.1)   # resolution increase
        #f_resolution.SetParLimits(2, 0.1, 5.)      # resolution turn-on position
        #f_resolution.SetParLimits(3, 0.01, 1.)     # resolution turn-on steepness
        
        g_resolution.Fit(f_resolution, 'RMS+')
        
        pid_params = (f_log_mean.GetParameter(0), f_log_mean.GetParameter(1), f_log_mean.GetParameter(2),
                        f_log_mean.GetParameter(3), f_log_mean.GetParameter(4), f_resolution.GetParameter(0),
                        f_resolution.GetParameter(1), f_resolution.GetParameter(2), f_resolution.GetParameter(3))
            
        particle_dir.cd()
        g_mean.Write()
        g_log_mean.Write()
        g_residuals.Write()
        g_resolution.Write()
        
        residual_correction = lambda betagamma: residual_params[0]*np.exp(-0.5*((betagamma-residual_params[1])/residual_params[2])**2) + \
            residual_params[3]*np.exp(-0.5*((betagamma-residual_params[4])/residual_params[5])**2)

        self.dataset['fExpTpcSignal'] = np_bethe_bloch(np.abs(self.dataset['fBetaGamma']), *pid_params[:5])
        self.dataset['fExpTpcSignalWithCorrections'] = np_bethe_bloch(np.abs(self.dataset['fBetaGamma']), *pid_params[:5]) + \
            residual_correction(np.abs(self.dataset['fBetaGamma']))
        
        compute_resolution = lambda betagamma: pid_params[5] #+ pid_params[6] * 1/ (1 + np.exp(- (betagamma-pid_params[7]) / pid_params[8]))
        compute_resolution_vectorised = np.vectorize(compute_resolution)
        self.dataset['fResolution'] = compute_resolution_vectorised(abs(self.dataset['fBetaGamma']))
        self.dataset['fNSigmaTPC'] = (self.dataset[self.cfg['dataset']['variable_names'][particle]['signal']] - self.dataset['fExpTpcSignal']) / (self.dataset['fExpTpcSignal'] * self.dataset['fResolution'])
        self.dataset['fNSigmaTPCWithCorrections'] = (self.dataset[self.cfg['dataset']['variable_names'][particle]['signal']] - self.dataset['fExpTpcSignalWithCorrections']) / (self.dataset['fExpTpcSignalWithCorrections'] * self.dataset['fResolution'])
        
        axis_spec_betagamma = AxisSpec(320, -8, 8, 'beta_gamma', ';#beta#gamma;d#it{E}/d#it{x} (a.u.)')
        axis_spec_tpcsignal = AxisSpec(100, 0, 2000, 'tpc_signal', ';#beta#gamma;d#it{E}/d#it{x} (a.u.)')
        axis_spec_nsigmatpc = AxisSpec(100, -5, 5, 'nsigma_tpc', ';#beta#gamma;n#sigma_{TPC}')
        axis_spec_clsize = AxisSpec(90, 0, 15., 'cl_size', ';#beta#gamma;#LT ITS cluster size (a.u.)#GT #times #LT cos#lambda#GT')

        h2_nsigmatpc = self.dataset.build_th2(f'{x_meta["var_name"]}', 'fNSigmaTPC', axis_spec_betagamma, axis_spec_nsigmatpc, title=f';{x_meta["axis_title"]};n#sigma_{{TPC}}')
        h2_nsigmatpc_with_corrections = self.dataset.build_th2(f'{x_meta["var_name"]}', 'fNSigmaTPCWithCorrections', axis_spec_betagamma, axis_spec_nsigmatpc, 
                                                               title=f';{x_meta["axis_title"]};n#sigma_{{TPC}} with corrections', name='h2_nsigmatpc_with_corrections')
        h2_exptpc = self.dataset.build_th2(f'{x_meta["var_name"]}', 'fExpTpcSignal', axis_spec_betagamma, axis_spec_tpcsignal)
        h2_tpc = self.dataset.build_th2(f'{x_meta["var_name"]}', self.cfg['dataset']['variable_names'][particle]['signal'], 
                                        axis_spec_betagamma, axis_spec_tpcsignal)

        f_fit_matter = TF1('f_fit_matter', BetheBloch, x_min, x_max, 5)
        f_fit_matter.SetParameters(*pid_params[:5])
        def BetheBlochAntimatter(x, *params):
            return BetheBloch(-x, *params)
        f_fit_antimatter = TF1('f_fit_antimatter', BetheBlochAntimatter, -x_max, -x_min, 5)
        f_fit_antimatter.SetParameters(*pid_params[:5])

        particle_dir.cd()
        
        h2_tpc.Write()
        h2_nsigmatpc.Write()
        h2_nsigmatpc_with_corrections.Write()
        h2_exptpc.Write('exp_tpc_signal')
        
        canvas = TCanvas('cNSigmaTPC', 'cNSigmaTPC', 800, 600)
        h2_tpc.Draw('colz')
        f_fit_matter.Draw('same')
        f_fit_antimatter.Draw('same')
        canvas.Write()
        
    