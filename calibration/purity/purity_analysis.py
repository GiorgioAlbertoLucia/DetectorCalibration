"""
Main script for purity analysis.
"""
import numpy as np
import pandas as pd
from ROOT import TFile, TH2F, RooRealVar, TF1, TCanvas

from torchic.core.graph import create_graph
from torchic.core.histogram import load_hist

import sys
sys.path.append('..')
from calibration.common.plot_utils import PdfHandler

from calibration.purity.purity_fitter import PurityFitter

class PurityAnalysis:
    """Main analysis class for purity extraction."""
    
    def __init__(self, input_file_path: str, output_file_path: str):
        """
        Initialize analysis.
        
        Parameters
        ----------
        input_file_path : str
            Path to input ROOT file with histograms
        output_file_path : str
            Path to output ROOT file for results
        """
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
    
    def run(self, outfile: TFile, particle: str, detector: str = 'TPC', config: dict = None):
        """
        Run full purity analysis for one particle/detector combination.
        
        Parameters
        ----------
        particle : str
            Particle type ('He3' or 'Had')
        detector : str
            Detector name ('TPC' or 'TOF')
        """
        print(f'\n{"="*60}')
        print(f'Running purity analysis: {particle} - {detector}')
        print(f'{"="*60}\n')
        print(f'Using config: {config}')
        
        h2_nsigma = self._load_histogram(particle, detector)
        if h2_nsigma is None:
            return
        
        particle_dir = outfile.mkdir(particle) if not outfile.GetDirectory(particle) else outfile.GetDirectory(particle)
        detector_dir = particle_dir.mkdir(detector) if not particle_dir.GetDirectory(detector) else particle_dir.GetDirectory(detector)
        
        fits_dir = detector_dir.mkdir('fits')
        bkg_fits_dir = detector_dir.mkdir('bkg_fits')
        
        fitter = PurityFitter(particle, detector, config)
        
        for sign in ['matter', 'antimatter']:
            print(f'\nProcessing {sign}...')
            self._analyze_sign(h2_nsigma, fitter, sign, 
                             fits_dir, bkg_fits_dir, detector_dir, 
                             particle, detector)
        
        detector_dir.cd()
        name_hist = f'h2_nsigma{particle}{detector}_preselection'
        h2_nsigma.Write(name_hist)
        
        print(f'\nResults saved to {self.output_file_path}')
    
    def _load_histogram(self, particle: str, detector: str):
        """Load input histogram from file."""
        particle_name = 'He3' if particle == 'He3' else 'Hadron'
        hist_name = f'he3-hadron-femto/QA/{particle}/h2Nsigma{particle_name}{detector}_preselection'
        
        try:
            h2 = load_hist(self.input_file_path, hist_name)
            h2.RebinY(2)  # Rebin for better statistics
            print(f'Loaded histogram: {hist_name}')
            print(f'  Entries: {h2.GetEntries():.0f}')
            return h2
        except Exception as e:
            print(f'Error loading histogram {hist_name}: {e}')
            return None
    
    def _analyze_sign(self, h2_nsigma: TH2F, fitter: PurityFitter, sign: str,
                     fits_dir, bkg_fits_dir, detector_dir, particle: str, detector: str):
        """Analyze one charge sign (matter or antimatter)."""
        
        pt_min, pt_max = self._get_pt_range(particle, sign, fitter.config)
        
        nsigma_range = fitter.config['fit_range']
        nsigma = RooRealVar('nsigma', f'#it{{n}}#sigma_{{{detector}}}', 
                           nsigma_range[0], nsigma_range[1])
        
        pdf_path = self.output_file_path.replace('.root', f'_{particle}_{detector}_{sign}.pdf')
        
        with PdfHandler(pdf_path) as pdf_canvas:
            fit_results = []
            
            for pt_bin in range(h2_nsigma.GetXaxis().FindBin(pt_min), 
                               h2_nsigma.GetXaxis().FindBin(pt_max)+1):
                
                pt = h2_nsigma.GetXaxis().GetBinCenter(pt_bin)
                pt_low = h2_nsigma.GetXaxis().GetBinLowEdge(pt_bin)
                pt_high = h2_nsigma.GetXaxis().GetBinLowEdge(pt_bin + 1)
                
                print(f'  pT bin: [{pt_low:.2f}, {pt_high:.2f}] GeV/c')
                
                h_nsigma = h2_nsigma.ProjectionY(
                    f'h_nsigma{detector}_{pt:.2f}', pt_bin, pt_bin, 'e'
                )
                
                frame, results = fitter.fit_slice(
                    h_nsigma, (pt_low, pt_high), nsigma,
                    fits_dir, bkg_fits_dir
                )
                
                if frame is None or results is None:
                    continue
                
                fit_results.append(results)
                
                frame.SetMinimum(1e2 if particle == 'Had' else 1.)
                frame.SetMaximum(h_nsigma.GetMaximum() * 1.5)
                pdf_canvas.draw_and_save(frame, logy=True)
                fits_dir.cd()
                pdf_canvas.canvas.Write()
                pdf_canvas.clear()
            
            if not fit_results:
                print(f'  Warning: No successful fits for {sign}')
                return
            
            # Compile results and create summary plots
            df_results = pd.DataFrame(fit_results)
            self._create_summary_plots(df_results, h2_nsigma, pt_min, pt_max,
                                      pdf_canvas, detector_dir, sign)
    
    def _get_pt_range(self, particle: str, sign: str, config: dict):
        """Get pT range based on particle type and charge sign."""
        tmp_pt_min, tmp_pt_max = config['pt_range'][0], config['pt_range'][1]
        
        if sign == 'antimatter':
            return -tmp_pt_max, -tmp_pt_min
        else:
            return tmp_pt_min, tmp_pt_max
    
    def _create_summary_plots(self, df: pd.DataFrame, h2_nsigma: TH2F,
                             pt_min: float, pt_max: float,
                             pdf_canvas: PdfHandler, detector_dir, sign: str):
        """Create summary plots of fit results."""
        
        df['pt_err'] = h2_nsigma.GetXaxis().GetBinWidth(1) / 2.
        
        # Signal integral
        g_signal = create_graph(df, 'pt', 'int_sig', 'pt_err', 0.,
                               name=f'g_signal_{sign}',
                               title=';#it{p}_{T} (GeV/c);N_{sig}')
        g_signal.SetMarkerStyle(20)
        pdf_canvas.draw_save_and_clear(g_signal, draw_option='ap', logy=False)
        
        # Signal mean
        g_mean = create_graph(df, 'pt', 'sig_mean', 'pt_err', 'sig_mean_err',
                             name=f'g_signal_mean_{sign}',
                             title=';#it{p}_{T} (GeV/c);#mu(n#sigma)')
        g_mean.SetMarkerStyle(20)
        
        fit_mean = TF1('fit_mean', '[0]', pt_min, pt_max)
        g_mean.Fit(fit_mean, 'RSM+')
        pdf_canvas.draw_save_and_clear(g_mean, draw_option='ap', logy=False)
        
        # Signal sigma
        g_sigma = create_graph(df, 'pt', 'sig_sigma', 'pt_err', 'sig_sigma_err',
                              name=f'g_signal_sigma_{sign}',
                              title=';#it{p}_{T} (GeV/c);#sigma(n#sigma)')
        g_sigma.SetMarkerStyle(20)
        
        fit_sigma = TF1('fit_sigma', 'pol0', pt_min, pt_max)
        g_sigma.Fit(fit_sigma, 'RSM+')
        pdf_canvas.draw_save_and_clear(g_sigma, draw_option='ap', logy=False)
        
        # Purity
        g_purity = create_graph(df, 'pt', 'purity', 'pt_err', 0.,
                               name=f'g_purity_{sign}',
                               title=';#it{p}_{T} (GeV/c);Purity')
        g_purity.SetMarkerStyle(20)
        g_purity.SetMinimum(0)
        g_purity.SetMaximum(1.1)
        pdf_canvas.draw_save_and_clear(g_purity, draw_option='ap', logy=False)
        
        # Save to ROOT file
        detector_dir.cd()
        g_signal.Write()
        g_mean.Write()
        g_sigma.Write()
        g_purity.Write()
        
        print(f'  Summary plots saved for {sign}')
