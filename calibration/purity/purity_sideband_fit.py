"""
Improved sideband fitting with ROOT TF1.
"""
import numpy as np
from ROOT import TF1, TCanvas, TPaveText, RooRealVar


class SidebandFitter:
    """Handles sideband fitting with TF1 before RooFit."""
    
    TF1_FUNCTIONS = {
        'exp': {
            'formula': '[0]*exp(-[1]*(x-[2]))',
            'npars': 3,
            'init': lambda: [100., 0.9, -5.],
            'alternative_init': lambda: [100., 0.2, -13.],
            'limits': [(0., 1e10), (0., 5.), (-20., 0.)],
            'alternative_limits': [(0., 1e10), (0., 1.), (-20., 0.)]
        },
        'gaus': {
            'formula': '[0]*exp(-0.5*((x-[1])/[2])**2)',
            'npars': 3,
            'init': lambda: [100., -5., 1.5],
            'limits': [(0., 1e10), (-10., -2.), (0.1, 5.)]
        },
        'pol0': {
            'formula': '[0]',
            'npars': 1,
            'init': lambda: [100.],
            'limits': [(0., 1e10)]
        },
        'pol1': {
            'formula': '[0] + [1]*x',
            'npars': 2,
            'init': lambda: [100., 0.],
            'limits': [(0., 1e10), (-1e4, 1e4)]
        },
        'pol2': {
            'formula': '[0] + [1]*x + [2]*x*x',
            'npars': 3,
            'init': lambda: [100., 0., 0.],
            'limits': [(0., 1e10), (-1e4, 1e4), (-1e3, 1e3)]
        },
        'gausexp': {
            'formula': '[0]*exp(-0.5*((x-[1])/[2])**2)*(x<[1]) + [0]*exp(-0.5*[3]*(x-[1]))*(x>=[1])',
            'npars': 4,
            'init': lambda: [100., -5., 1.5, 2.],
            'limits': [(0., 1e10), (-10., -3.), (1., 5.), (0.1, 10.)]
        }
    }
    
    COMPOSITE_FUNCTIONS = {
        'pol1+gausexp': ['pol1', 'gausexp'],
        'pol2+gausexp': ['pol2', 'gausexp'],
        'gausexp+exp': ['gausexp', 'exp'],
        'exp+gaus': ['exp', 'gaus'],
        'pol1+exp': ['pol1', 'exp'],
        'pol0+gausexp': ['pol0', 'gausexp'],
        'exp+exp': ['exp', 'exp'],
        'gausexp+gausexp': ['gausexp', 'gausexp']
    }
    
    @classmethod
    def create_tf1(cls, bkg_name: str, fit_range: tuple):
        """
        Create TF1 function for the given background type.
        
        Parameters
        ----------
        bkg_name : str
            Name of background function
        fit_range : tuple
            (min, max) range for fitting
            
        Returns
        -------
        TF1
            Configured TF1 function
        """
        if bkg_name in cls.COMPOSITE_FUNCTIONS:
            return cls._create_composite_tf1(bkg_name, fit_range)
        elif bkg_name in cls.TF1_FUNCTIONS:
            return cls._create_simple_tf1(bkg_name, fit_range)
        else:
            raise ValueError(f'Unsupported background function: {bkg_name}')
    
    @classmethod
    def _create_simple_tf1(cls, bkg_name: str, fit_range: tuple):
        """Create a simple (non-composite) TF1."""
        func_info = cls.TF1_FUNCTIONS[bkg_name]
        tf1 = TF1(f'tf1_{bkg_name}', func_info['formula'], fit_range[0], fit_range[1])
        
        init_pars = func_info['init']()
        for i, val in enumerate(init_pars):
            tf1.SetParameter(i, val)
        
        for i, lim in enumerate(func_info['limits']):
            tf1.SetParLimits(i, lim[0], lim[1])
        
        return tf1
    
    @classmethod
    def _create_composite_tf1(cls, bkg_name: str, fit_range: tuple):
        """Create a composite TF1 from multiple functions."""
        components = cls.COMPOSITE_FUNCTIONS[bkg_name]
        
        formulas = []
        all_init = []
        all_limits = []
        par_offset = 0
        
        for icomp, comp in enumerate(components):
            func_info = cls.TF1_FUNCTIONS[comp]
            formula = func_info['formula']
            
            for i in range(func_info['npars']):
                formula = formula.replace(f'[{i}]', f'[{par_offset + i}]')
            
            duplicate_formula = formula in formulas[:icomp] or any(formula == 'exp' and 'exp' in _formula for _formula in formula[:icomp])
            init = func_info['alternative_init']() if duplicate_formula and 'alternative_init' in func_info \
                                                    else func_info['init']()
            limits = func_info['alternative_limits'] if duplicate_formula and 'alternative_limits' in func_info \
                                                    else func_info['limits']

            formulas.append(f'({formula})')
            all_init.extend(init)
            all_limits.extend(limits)
            par_offset += func_info['npars']
        
        full_formula = ' + '.join(formulas)
        tf1 = TF1(f'tf1_{bkg_name}', full_formula, fit_range[0], fit_range[1])
        
        for i, val in enumerate(all_init):
            tf1.SetParameter(i, val)
        
        for i, lim in enumerate(all_limits):
            tf1.SetParLimits(i, lim[0], lim[1])
        
        return tf1
    
    @classmethod
    def fit_sidebands(cls, h_data, bkg_name: str, fit_range: tuple, 
                     sig_window: tuple, outdir, pt: float):
        """
        Fit sidebands with TF1 to get background shape.
        
        Parameters
        ----------
        h_data : TH1
            Data histogram
        bkg_name : str
            Background function name
        fit_range : tuple
            Full fit range (min, max)
        sig_window : tuple
            Signal window (min, max) to exclude
        outdir : TDirectory
            Output directory for plots
        pt : float
            Transverse momentum value (for labeling)
            
        Returns
        -------
        tuple
            (tf1_function, chi2_ndf, integral_in_fit_range)
        """

        h_sidebands = h_data.Clone(f'{h_data.GetName()}_sidebands')
        x_low_max = h_sidebands.GetXaxis().FindBin(sig_window[0])
        x_high_min = h_sidebands.GetXaxis().FindBin(sig_window[1])
        print(f'{sig_window=}')
        
        for ibin in range(x_low_max, x_high_min + 1):
            h_sidebands.SetBinContent(ibin, 0.)
            h_sidebands.SetBinError(ibin, 0.)
        
        tf1_bkg = cls.create_tf1(bkg_name, fit_range)
        
        fit_result = h_sidebands.Fit(tf1_bkg, 'RQNS')  # First attempt: quiet
        
        if fit_result.Status() != 0:
            fit_result = h_sidebands.Fit(tf1_bkg, 'RMNS')  # Second attempt: Minuit
        
        if fit_result.Status() != 0:
            fit_result = h_sidebands.Fit(tf1_bkg, 'RMNS', '', fit_range[0], fit_range[1])
        
        chi2_ndf = tf1_bkg.GetChisquare() / tf1_bkg.GetNDF() if tf1_bkg.GetNDF() > 0 else 1e10
        
        bin_width = h_data.GetBinWidth(1)
        integral = tf1_bkg.Integral(fit_range[0], fit_range[1]) / bin_width
        
        cls._save_fit_plot(h_sidebands, tf1_bkg, chi2_ndf, integral, 
                          bkg_name, pt, outdir)
        
        return tf1_bkg, chi2_ndf, integral
    
    @classmethod
    def _save_fit_plot(cls, h_sidebands, tf1_bkg, chi2_ndf, integral, 
                      bkg_name: str, pt: float, outdir):
        """Save diagnostic plot of sideband fit."""
        canvas = TCanvas(f'sidebands_{bkg_name}_fit_{pt:.2f}', '', 800, 600)
        canvas.SetLogy(True)
        
        h_sidebands.SetMinimum(0.1)
        h_sidebands.SetLineColor(1)
        h_sidebands.SetMarkerStyle(20)
        h_sidebands.SetMarkerSize(0.8)
        h_sidebands.Draw('E')
        
        tf1_bkg.SetLineColor(2)
        tf1_bkg.SetLineWidth(2)
        tf1_bkg.Draw('same')
        
        # Add text with fit results
        text = TPaveText(0.6, 0.7, 0.88, 0.88, 'ndc')
        text.SetFillColor(0)
        text.SetBorderSize(1)
        text.SetTextAlign(12)
        text.AddText(f'Function: {bkg_name}')
        text.AddText(f'#chi^{{2}}/ndf = {chi2_ndf:.2f}')
        text.AddText(f'N_{{bkg}} = {integral:.0f}')
        text.AddText(f'p_{{T}} = {pt:.2f} GeV/c')
        text.Draw('same')
        
        outdir.cd()
        canvas.Write()
        canvas.Close()
    
    @classmethod
    def transfer_parameters(cls, tf1_func, bkg_name: str, bkg_pars: dict, fit_range: tuple):
        """
        Transfer fitted parameters from TF1 to RooFit variables.
        
        Parameters
        ----------
        tf1_func : TF1
            Fitted TF1 function
        bkg_name : str
            Background function name
        bkg_pars : dict
            Dictionary of RooRealVar parameters to update
        fit_range : tuple
            Fit range for normalization calculation
            
        Returns
        -------
        list of RooRealVar or RooRealVar
            Normalization parameters
        """
        bin_width = 0.1  # Assume standard bin width, can be passed if needed
        
        if bkg_name == 'exp':
            bkg_pars['alpha'].setVal(tf1_func.GetParameter(1))
            bkg_pars['offset'].setVal(tf1_func.GetParameter(2))
            integral = tf1_func.Integral(fit_range[0], fit_range[1]) / bin_width
            return RooRealVar('bkg_normalisation', 'N_bkg', integral, 0., integral*10)
            
        elif bkg_name == 'gaus':
            bkg_pars['mean'].setVal(tf1_func.GetParameter(1))
            bkg_pars['sigma'].setVal(tf1_func.GetParameter(2))
            integral = tf1_func.Integral(fit_range[0], fit_range[1]) / bin_width
            return RooRealVar('bkg_normalisation', 'N_bkg', integral, 0., integral*10)
            
        elif bkg_name == 'pol1':
            bkg_pars['p0'].setVal(tf1_func.GetParameter(0))
            bkg_pars['p1'].setVal(tf1_func.GetParameter(1))
            integral = tf1_func.Integral(fit_range[0], fit_range[1]) / bin_width
            return RooRealVar('bkg_normalisation', 'N_bkg', integral, 0., integral*10)
            
        elif bkg_name == 'pol2':
            bkg_pars['p0'].setVal(tf1_func.GetParameter(0))
            bkg_pars['p1'].setVal(tf1_func.GetParameter(1))
            bkg_pars['p2'].setVal(tf1_func.GetParameter(2))
            integral = tf1_func.Integral(fit_range[0], fit_range[1]) / bin_width
            return RooRealVar('bkg_normalisation', 'N_bkg', integral, 0., integral*10)
            
        elif bkg_name == 'gausexp':
            bkg_pars['mean'].setVal(tf1_func.GetParameter(1))
            bkg_pars['sigma'].setVal(tf1_func.GetParameter(2))
            if 'rlife' in bkg_pars:
                bkg_pars['rlife'].setVal(tf1_func.GetParameter(3))
            integral = tf1_func.Integral(fit_range[0], fit_range[1]) / bin_width
            return RooRealVar('bkg_normalisation', 'N_bkg', integral, 0., integral*10)
        
        elif bkg_name in cls.COMPOSITE_FUNCTIONS:
            return cls._transfer_composite_parameters(tf1_func, bkg_name, bkg_pars, fit_range, bin_width)
        
        return None
    
    @classmethod
    def _transfer_composite_parameters(cls, tf1_func, bkg_name: str, bkg_pars: dict, 
                                      fit_range: tuple, bin_width: float):
        """Transfer parameters for composite backgrounds."""
        components = cls.COMPOSITE_FUNCTIONS[bkg_name]
        par_offset = 0
        normalizations = []

        same_function = len(components) != len(set(components))
        
        # Extract parameters for each component
        for i, comp in enumerate(components):
            func_info = cls.TF1_FUNCTIONS[comp]
            npars = func_info['npars']

            suffix = f'_{i}' if same_function else ''
            
            if comp == 'pol1':
                bkg_pars[f'p0{suffix}'].setVal(tf1_func.GetParameter(par_offset))
                bkg_pars[f'p1{suffix}'].setVal(tf1_func.GetParameter(par_offset + 1))
            elif comp == 'pol2':
                bkg_pars[f'p0{suffix}'].setVal(tf1_func.GetParameter(par_offset))
                bkg_pars[f'p1{suffix}'].setVal(tf1_func.GetParameter(par_offset + 1))
                bkg_pars[f'p2{suffix}'].setVal(tf1_func.GetParameter(par_offset + 2))
            elif comp == 'pol0':
                pass  # No parameters to transfer
            elif comp == 'exp':
                bkg_pars[f'alpha{suffix}'].setVal(tf1_func.GetParameter(par_offset + 1))
                bkg_pars[f'offset{suffix}'].setVal(tf1_func.GetParameter(par_offset + 2))
            elif comp == 'gaus':
                if 'mean_bkg' in bkg_pars:
                    bkg_pars[f'mean_bkg{suffix}'].setVal(tf1_func.GetParameter(par_offset + 1))
                    bkg_pars[f'sigma_bkg{suffix}'].setVal(tf1_func.GetParameter(par_offset + 2))
                elif 'mean' in bkg_pars:
                    bkg_pars[f'mean{suffix}'].setVal(tf1_func.GetParameter(par_offset + 1))
                    bkg_pars[f'sigma{suffix}'].setVal(tf1_func.GetParameter(par_offset + 2))
            elif comp == 'gausexp':
                if 'mean_bkg' in bkg_pars:
                    bkg_pars[f'mean_bkg{suffix}'].setVal(tf1_func.GetParameter(par_offset + 1))
                    bkg_pars[f'sigma_bkg{suffix}'].setVal(tf1_func.GetParameter(par_offset + 2))
                    bkg_pars[f'bkg_rlife{suffix}'].setVal(tf1_func.GetParameter(par_offset + 3))
            
            # Calculate integral for this component - approximate as total/n_components
            total_integral = tf1_func.Integral(fit_range[0], fit_range[1]) / bin_width
            comp_integral = total_integral / len(components)
            
            norm = RooRealVar(f'bkg_normalisation_{i}', f'N_bkg_{i}', 
                            comp_integral, 0., total_integral*5)
            normalizations.append(norm)
            
            par_offset += npars
        
        return normalizations if len(normalizations) > 1 else normalizations[0]
    