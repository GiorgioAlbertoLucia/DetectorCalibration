"""
Main fitting logic for purity extraction.
"""
import numpy as np
import pandas as pd
from ROOT import RooRealVar, RooAddPdf, RooDataHist, TPaveText

from calibration.purity.purity_models import SignalModel, BackgroundModel
from calibration.purity.purity_sideband_fit import SidebandFitter


class PurityFitter:
    """Handles the full fitting procedure for purity extraction."""
    
    PARTICLE_CONFIG = {
        'He3': {
            'signal_model': 'gausdexp',
            'bkg_models': ['pol1', 'pol0', 'exp', 'gausexp'],
            'signal_window': (-2, 4),
            'fit_range': (-6, 4),
            'nsigma_integral': (-2, 2),
            'info_box_position': (0.35, 0.2, 0.65, 0.48),
        },
        'Had': {
            'signal_model': 'gausdexp', #'gausdexp',
            'bkg_models': ['pol1', 
                           #'pol2', 
                           'exp', 
                           #'pol2+gausexp', 
                           #'pol1+exp',
                           #'pol0+exp',
                           #'pol1+gausexp'
                           'exp+exp'
                           ],
            'signal_window': (-4, 5.5),
            'fit_range': (-6, 5),
            'nsigma_integral': (-2, 2),
            'info_box_position': (0.35, 0.2, 0.65, 0.48),
        }
    }
    
    def __init__(self, particle: str, detector: str = 'TPC', config:dict=None):
        """
        Initialize fitter.
        
        Parameters
        ----------
        particle : str
            Particle type ('He3' or 'Had')
        detector : str
            Detector name ('TPC' or 'TOF')
        """
        self.particle = particle
        self.detector = detector
        self.config = config if config is not None else self.PARTICLE_CONFIG.get(particle)
        
        if self.config is None:
            raise ValueError(f'Unknown particle type: {particle}')
    
    def fit_slice(self, hist, pt_range: tuple, x_variable: RooRealVar, 
                  outdir_main, outdir_bkg):
        """
        Fit a single pT slice.
        
        Parameters
        ----------
        hist : TH1
            Histogram to fit
        pt_range : tuple
            (pt_low, pt_high) for this slice
        x_variable : RooRealVar
            Observable variable
        outdir_main : TDirectory
            Directory for main fit results
        outdir_bkg : TDirectory
            Directory for background fit diagnostics
            
        Returns
        -------
        tuple
            (roofit_frame, fit_results_dict)
        """
        pt = 0.5 * (pt_range[0] + pt_range[1])
        
        if hist.GetEntries() < 100:
            print(f'Warning: Not enough entries ({hist.GetEntries()}) at pT={pt:.2f}')
            return None, None
        
        signal_pdf, signal_pars = SignalModel.create(
            x_variable, self.config['signal_model'], self.particle
        )
        
        best_bkg = self._select_best_background(
            hist, x_variable, pt, outdir_bkg
        )
        
        if best_bkg is None:
            print(f'Warning: No valid background model found at pT={pt:.2f}')
            return None, None
        
        bkg_pdf, bkg_pars, bkg_norm = best_bkg
        #for param in bkg_pars.values():
        #    param.setConstant(True)
        
        signal_norm = RooRealVar('signal_normalisation', 'N_sig', 
                                hist.GetEntries() * 0.5, 0., hist.GetEntries() * 2)
        
        if isinstance(bkg_pdf, list):
            model = RooAddPdf('model', 'signal + bkg', 
                            [signal_pdf] + bkg_pdf, 
                            [signal_norm] + bkg_norm)
        else:
            model = RooAddPdf('model', 'signal + bkg', 
                            [signal_pdf, bkg_pdf], 
                            [signal_norm, bkg_norm])
        
        dh = RooDataHist(f'dh_{pt:.2f}', 'dh', [x_variable], Import=hist)
        fit_result = model.fitTo(dh, PrintLevel=-1, Save=True)
        
        frame = self._create_fit_frame(x_variable, dh, model, signal_pdf, 
                                      bkg_pdf, pt_range, signal_pars, bkg_pars)
        fit_results = self._calculate_purity(x_variable, signal_pdf, bkg_pdf,
                                            signal_norm, bkg_norm, signal_pars, pt)
        
        outdir_main.cd()
        
        return frame, fit_results
    
    def _select_best_background(self, hist, x_variable, pt, outdir_bkg):
        """
        Try multiple background models and select the best one.
        
        Returns
        -------
        tuple or None
            (bkg_pdf, bkg_pars, bkg_norm) for best model, or None if all fail
        """
        results = []
        
        for bkg_pt_min, bkg_pt_max, bkg_name in zip(self.config['bkg_pt_min'], self.config['bkg_pt_max'], self.config['bkg_models']):
            try:
                if np.abs(pt) < bkg_pt_min:
                    continue
                if np.abs(pt) > bkg_pt_max:
                    continue

                print(f'  Trying {bkg_name}...', end=' ')
                
                tf1_func, chi2_ndf, integral = SidebandFitter.fit_sidebands(
                    hist, bkg_name, 
                    self.config['fit_range'],
                    self.config['signal_window'],
                    outdir_bkg, pt
                )
                
                if chi2_ndf > 1e8 or chi2_ndf < 0 or np.isnan(chi2_ndf):
                    print(f'chi2/ndf = {chi2_ndf:.2f} - REJECTED')
                    continue
                
                bkg_pdf, bkg_pars = BackgroundModel.create(
                    x_variable, bkg_name, particle=self.particle, suffix=f'_{bkg_name}'
                )
                
                bkg_norm = SidebandFitter.transfer_parameters(
                    tf1_func, bkg_name, bkg_pars, self.config['fit_range']
                )
                
                if bkg_norm is None:
                    if isinstance(bkg_pdf, list):
                        bkg_norm = [
                            RooRealVar(f'bkg_normalisation_{i}', f'N_bkg_{i}', 
                                     integral/len(bkg_pdf), 0., integral*10)
                            for i in range(len(bkg_pdf))
                        ]
                    else:
                        bkg_norm = RooRealVar('bkg_normalisation', 'N_bkg', 
                                            integral, 0., integral*10)
                
                results.append((chi2_ndf, bkg_pdf, bkg_pars, bkg_norm, bkg_name))
                print(f'chi2/ndf = {chi2_ndf:.2f}')
                
            except Exception as e:
                print(f'ERROR: {e}')
                continue
        
        if not results:
            print('  No valid background models found!')
            return None
        
        results.sort(key=lambda x: x[0])
        best = results[0]
        print(f'  --> SELECTED: {best[4]} (chi2/ndf = {best[0]:.2f})')
        
        return best[1], best[2], best[3]
    
    def _create_fit_frame(self, x_variable, data_hist, model, signal_pdf, 
                         bkg_pdf, pt_range, signal_pars, bkg_pars):
        """Create RooFit frame with all components."""
        frame = x_variable.frame(
            Title=f'{pt_range[0]:.2f} < #it{{p}}_{{T}} < {pt_range[1]:.2f} GeV/#it{{c}}'
        )
        
        data_hist.plotOn(frame, Name='data')
        model.plotOn(frame, LineColor=2, LineWidth=2, Name='model')
        
        model.plotOn(frame, Components={signal_pdf}, 
                    LineColor=3, LineStyle=2, LineWidth=2, Name='signal')
        if isinstance(bkg_pdf, list):
            for i, bkg in enumerate(bkg_pdf):
                model.plotOn(frame, Components={bkg}, 
                           LineColor=4+i, LineStyle=2, LineWidth=2, 
                           Name=f'bkg_{i}')
        else:
            model.plotOn(frame, Components={bkg_pdf}, 
                       LineColor=4, LineStyle=2, LineWidth=2, Name='background')
        
        text = TPaveText(*self.config['info_box_position'], 'ndc')
        text.SetFillColor(0)
        text.SetBorderSize(0)
        text.SetTextAlign(12)
        text.SetTextSize(0.03)
        
        text.AddText(f'#mu = {signal_pars["mean"].getVal():.3f} #pm {signal_pars["mean"].getError():.3f}')
        text.AddText(f'#sigma = {signal_pars["sigma"].getVal():.3f} #pm {signal_pars["sigma"].getError():.3f}')
        
        if 'rlife0' in signal_pars:
            text.AddText(f'rlife0 = {signal_pars["rlife0"].getVal():.3f} #pm {signal_pars["rlife0"].getError():.3f}')
            text.AddText(f'rlife1 = {signal_pars["rlife1"].getVal():.3f} #pm {signal_pars["rlife1"].getError():.3f}')
        elif 'rlife' in signal_pars:
            text.AddText(f'rlife = {signal_pars["rlife"].getVal():.3f}')
        
        text.AddText(f'#chi^{{2}}/ndf = {frame.chiSquare():.2f}')

        for param_name, param in bkg_pars.items():
            text.AddText(f'{param_name} = {param.getVal():.3f} #pm {param.getError():.3f}')
        
        frame.addObject(text)
        
        return frame
    
    def _calculate_purity(self, x_variable, signal_pdf, bkg_pdf, 
                         signal_norm, bkg_norm, signal_pars, pt):
        """Calculate signal integral and purity."""
        x_variable.setRange('integral_range', 
                          self.config['nsigma_integral'][0],
                          self.config['nsigma_integral'][1])
        
        int_signal = (signal_pdf.createIntegral(x_variable, x_variable, 'integral_range').getVal() 
                     * signal_norm.getVal())
        
        int_bkg = 0.
        if isinstance(bkg_pdf, list):
            for pdf, norm in zip(bkg_pdf, bkg_norm):
                int_bkg += (pdf.createIntegral(x_variable, x_variable, 'integral_range').getVal() 
                           * norm.getVal())
        else:
            int_bkg = (bkg_pdf.createIntegral(x_variable, x_variable, 'integral_range').getVal() 
                      * bkg_norm.getVal())
        
        purity = int_signal / (int_signal + int_bkg) if (int_signal + int_bkg) > 0 else 0.
        
        results = {
            'pt': np.abs(pt),
            'int_sig': int_signal,
            'int_bkg': int_bkg,
            'sig_mean': signal_pars['mean'].getVal(),
            'sig_sigma': signal_pars['sigma'].getVal(),
            'sig_mean_err': signal_pars['mean'].getError(),
            'sig_sigma_err': signal_pars['sigma'].getError(),
            'purity': purity
        }
        
        return results
    