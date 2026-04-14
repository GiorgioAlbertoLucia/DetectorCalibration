"""
PDF model definitions for signal and background fitting.
"""
from ROOT import RooRealVar, RooCrystalBall, RooGenericPdf, RooGaussian
from torchic.roopdf import RooGausExp, RooGausDExp


class SignalModel:
    """Factory for signal PDF models."""
    
    @staticmethod
    def create(x: RooRealVar, function: str = 'gausdexp', particle: str = 'He3'):
        """
        Create a signal model.
        
        Parameters
        ----------
        x : RooRealVar
            Observable variable
        function : str
            Type of signal function ('crystalball', 'crystalballdb', 'gausexp', 'gausdexp')
        particle : str
            Particle type for parameter tuning
            
        Returns
        -------
        tuple
            (signal_pdf, signal_parameters_dict)
        """
        if function == 'crystalball':
            return SignalModel._create_crystalball(x)
        elif function == 'crystalballdb':
            return SignalModel._create_crystalball_double(x)
        elif function == 'gausexp':
            return SignalModel._create_gausexp(x)
        elif function == 'gausdexp':
            return SignalModel._create_gausdexp(x, particle)
        else:
            raise ValueError(f'Unknown signal function: {function}')
    
    @staticmethod
    def _create_crystalball(x):
        pars = {
            'mean': RooRealVar('mean', '#mu', 0.5, -1.5, 1.),
            'sigma': RooRealVar('sigma', '#sigma', 0.8, 0.3, 1),
            'aL': RooRealVar('aL', '#alpha', 1., 0.7, 10.),
            'nL': RooRealVar('nL', 'n', 1., 0.1, 30.)
        }
        pdf = RooCrystalBall('signal', 'signal', x, 
                            pars['mean'], pars['sigma'], 
                            pars['aL'], pars['nL'], doubleSided=True)
        return pdf, pars
    
    @staticmethod
    def _create_crystalball_double(x):
        pars = {
            'mean': RooRealVar('mean', '#mu', 0., -1.5, 1.5),
            'sigma': RooRealVar('sigma', '#sigma', 1, 0.3, 2),
            'aL': RooRealVar('aL', '#alpha_{L}', 6., 0.7, 10.),
            'nL': RooRealVar('nL', 'n_{L}', 1., 0.1, 30.),
            'aR': RooRealVar('aR', '#alpha_{R}', 6., 0.7, 10.),
            'nR': RooRealVar('nR', 'n_{R}', 1., 0.1, 30.)
        }
        pdf = RooCrystalBall('signal', 'signal', x, 
                            pars['mean'], pars['sigma'], 
                            pars['aL'], pars['nL'],
                            pars['aR'], pars['nR'])
        return pdf, pars
    
    @staticmethod
    def _create_gausexp(x):
        pars = {
            'mean': RooRealVar('mean', '#mu', 0.5, -1.2, 1.3),
            'sigma': RooRealVar('sigma', '#sigma', 0.8, 0.3, 1.3),
            'rlife': RooRealVar('rlife', 'rlife', 1.1, 0., 10.),
        }
        pdf = RooGausExp('signal', 'signal', x, *pars.values())
        return pdf, pars
    
    @staticmethod
    def _create_gausdexp(x, particle):
        # Adjust parameters based on particle type
        if particle == 'He3':
            mean_init, mean_min, mean_max = 0.5, -1.2, 1.3
            sigma_init, sigma_min, sigma_max = 0.8, 0.3, 1.3
        else:  # Had
            mean_init, mean_min, mean_max = 0.5, -1.0, 2.0
            sigma_init, sigma_min, sigma_max = 1.3, 0.5, 2.5
            
        pars = {
            'mean': RooRealVar('mean', '#mu', mean_init, mean_min, mean_max),
            'sigma': RooRealVar('sigma', '#sigma', sigma_init, sigma_min, sigma_max),
            'rlife0': RooRealVar('rlife0', 'rlife0', -0.8, -1.2, -0.5),
            'rlife1': RooRealVar('rlife1', 'rlife1', 2., 0.5, 1.2),
        }
        pdf = RooGausDExp('signal', 'signal', x, *pars.values())
        return pdf, pars


class BackgroundModel:
    """Factory for background PDF models."""
    
    @staticmethod
    def create(x: RooRealVar, function: str = 'pol1', particle:str = 'He3', suffix: str = ''):
        """
        Create a background model.
        
        Parameters
        ----------
        x : RooRealVar
            Observable variable
        function : str
            Type of background function
        suffix : str
            Suffix for PDF name
            
        Returns
        -------
        tuple
            (background_pdf or list of pdfs, parameters_dict)
        """
        if function == 'exp':
            return BackgroundModel._create_exp(x, particle, suffix)
        elif function == 'gaus':
            return BackgroundModel._create_gaus(x, particle, suffix)
        elif function == 'gausexp':
            return BackgroundModel._create_gausexp(x, particle, suffix)
        elif function == 'pol0':
            return BackgroundModel._create_pol0(x, particle, suffix)
        elif function == 'pol1':
            return BackgroundModel._create_pol1(x, particle, suffix)
        elif function == 'pol2':
            return BackgroundModel._create_pol2(x, particle, suffix)
        elif function == 'pol1+gausexp':
            return BackgroundModel._create_pol1_gausexp(x, particle, suffix)
        elif function == 'pol2+gausexp':
            return BackgroundModel._create_pol2_gausexp(x, particle, suffix)
        elif function == 'gausexp+exp':
            return BackgroundModel._create_gausexp_exp(x, particle, suffix)
        elif function == 'exp+gaus':
            return BackgroundModel._create_exp_gaus(x, particle, suffix)
        elif function == 'pol1+exp':
            return BackgroundModel._create_pol1_exp(x, particle, suffix)
        elif function == 'pol0+gausexp':
            return BackgroundModel._create_pol0_gausexp(x, particle, suffix)
        elif function == 'exp+exp':
            return BackgroundModel._create_exp_exp(x, particle, suffix)
        else:
            raise ValueError(f'Unknown background function: {function}')
    
    @staticmethod
    def _create_exp(x, particle, suffix):
        pars = {
            'alpha': RooRealVar('alpha', 'alpha', 1., 0.01, 100.),
            'offset': RooRealVar('offset', 'offset', -5., -20., 0.)
        }
        pdf = RooGenericPdf(f'bkg{suffix}', 'bkg', 
                           f'exp(-alpha * ({x.GetName()} - offset))', 
                           [x, *pars.values()])
        return pdf, pars
    
    @staticmethod
    def _create_gaus(x, particle, suffix):
        pars = {
            'mean': RooRealVar('mean_gaus', 'mean_gaus', -4., -10, -2),
            'sigma': RooRealVar('sigma_gaus', 'sigma_gaus', 1.0, 0.1, 5.)
        }
        if particle == 'He3':
            pars['mean'].setVal(-5.)
            pars['mean'].setRange(-10., -2.5)
        pdf = RooGaussian(f'bkg{suffix}', 'bkg', x, *pars.values())
        return pdf, pars
    
    @staticmethod
    def _create_gausexp(x, particle, suffix):
        pars = {
            'mean': RooRealVar('bkg_mean', 'mean', -4., -10, -2),
            'sigma': RooRealVar('bkg_sigma', 'sigma', 1.0, 0.1, 5.),
            'rlife': RooRealVar('bkg_rlife', 'rlife', 2., 0.5, 10.),
        }
        if particle == 'He3':
            pars['mean'].setVal(-5.)
            pars['mean'].setRange(-10., -2.5)
        pdf = RooGausExp(f'bkg{suffix}', f'bkg{suffix}', x, *pars.values())
        return pdf, pars
    
    @staticmethod
    def _create_pol0(x, particle, suffix):
        pars = {}
        pdf = RooGenericPdf(f'bkg{suffix}', 'bkg', '1', [x])
        return pdf, pars
    
    @staticmethod
    def _create_pol1(x, particle, suffix):
        pars = {
            'p0': RooRealVar('p0', 'c_{0}', 1., 0., 1e6),
            'p1': RooRealVar('p1', 'c_{1}', 0., -1e4, 1e4),
        }
        pdf = RooGenericPdf(f'bkg{suffix}', 'bkg', 
                           f'p0 + p1 * {x.GetName()}', 
                           [x, *pars.values()])
        return pdf, pars
    
    @staticmethod
    def _create_pol2(x, particle, suffix):
        pars = {
            'p0': RooRealVar('p0', 'c_{0}', 1., 0., 1e6),
            'p1': RooRealVar('p1', 'c_{1}', 0., -1e4, 1e4),
            'p2': RooRealVar('p2', 'c_{2}', 0., -1e4, 1e4),
        }
        pdf = RooGenericPdf(f'bkg{suffix}', 'bkg', 
                           f'p0 + p1*{x.GetName()} + p2*{x.GetName()}*{x.GetName()}', 
                           [x, *pars.values()])
        return pdf, pars
    
    @staticmethod
    def _create_pol1_gausexp(x, particle, suffix):
        pars = {
            'p0': RooRealVar('p0', 'c_{0}', 1., 0., 1e6),
            'p1': RooRealVar('p1', 'c_{1}', 0., -1e4, 1e4),
            'mean': RooRealVar('mean_bkg', 'mean_bkg', -4., -10, -2),
            'sigma': RooRealVar('sigma_bkg', 'sigma_bkg', 1.0, 0.1, 5.),
            'rlife': RooRealVar('bkg_rlife', 'rlife', 2., 0.5, 10.),
        }
        if particle == 'He3':
            pars['mean'].setVal(-5.)
            pars['mean'].setRange(-10., -2.5)
        pol1 = RooGenericPdf(f'bkg{suffix}_pol1', 'bkg', 
                            f'p0 + p1 * {x.GetName()}', 
                            [x, pars['p0'], pars['p1']])
        gausexp = RooGausExp(f'bkg{suffix}_gausexp', 'bkg', x, 
                            pars['mean'], pars['sigma'], pars['rlife'])
        return [pol1, gausexp], pars
    
    @staticmethod
    def _create_pol2_gausexp(x, particle, suffix):
        pars = {
            'p0': RooRealVar('p0', 'c_{0}', 1., 0., 1e6),
            'p1': RooRealVar('p1', 'c_{1}', 0., -1e4, 1e4),
            'p2': RooRealVar('p2', 'c_{2}', 0., -1e4, 1e4),
            'mean': RooRealVar('mean_bkg', 'mean_bkg', -4., -10, -2),
            'sigma': RooRealVar('sigma_bkg', 'sigma_bkg', 1.0, 0.1, 5.),
            'rlife': RooRealVar('bkg_rlife', 'rlife', 2., 0.5, 10.),
        }
        if particle == 'He3':
            pars['mean'].setVal(-5.)
            pars['mean'].setRange(-10., -2.5)
        pol2 = RooGenericPdf(f'bkg{suffix}_pol2', 'bkg', 
                            f'p0 + p1*{x.GetName()} + p2*{x.GetName()}*{x.GetName()}', 
                            [x, pars['p0'], pars['p1'], pars['p2']])
        gausexp = RooGausExp(f'bkg{suffix}_gausexp', 'bkg', x, 
                            pars['mean'], pars['sigma'], pars['rlife'])
        return [pol2, gausexp], pars
    
    @staticmethod
    def _create_gausexp_exp(x, particle, suffix):
        pars = {
            'alpha': RooRealVar('alpha', 'alpha', 0.14, 0.001, 0.3),
            'offset': RooRealVar('offset', 'offset', -5., -20., 0.),
            'mean': RooRealVar('mean_bkg', 'mean_bkg', -5., -10, -2),
            'sigma': RooRealVar('sigma_bkg', 'sigma_bkg', 1.0, 0.1, 5.),
            'rlife': RooRealVar('bkg_rlife', 'rlife', 2., 0.5, 10.),
        }
        if particle == 'He3':
            pars['mean'].setVal(-5.)
            pars['mean'].setRange(-10., -2.5)
        exp = RooGenericPdf(f'bkg{suffix}_exp', 'bkg', 
                           f'exp(-alpha * ({x.GetName()} - offset))', 
                           [x, pars['alpha'], pars['offset']])
        gausexp = RooGausExp(f'bkg{suffix}_gausexp', 'bkg', x, 
                            pars['mean'], pars['sigma'], pars['rlife'])
        return [gausexp, exp], pars

    @staticmethod
    def _create_exp_exp(x, particle, suffix):
        pars = {
            'alpha_0': RooRealVar('alpha_0', 'alpha_0', 0.05, 0., 0.15),
            'offset_0': RooRealVar('offset_0', 'offset_0', -5., -20., 0.),
            'alpha_1': RooRealVar('alpha_1', 'alpha_1', 1., 0.9, 2.),
            'offset_1': RooRealVar('offset_1', 'offset_1', -5., -20., 0.),
        }
        exp2 = RooGenericPdf(f'bkg{suffix}_exp_0', 'bkg', 
                           f'exp(-alpha_0 * ({x.GetName()} - offset_0))', 
                           [x, pars['alpha_0'], pars['offset_0']])
        exp1 = RooGenericPdf(f'bkg{suffix}_exp_1', 'bkg', 
                           f'exp(-alpha_1 * ({x.GetName()} - offset_1))', 
                           [x, pars['alpha_1'], pars['offset_1']])
        
        return [exp1, exp2], pars
    
    @staticmethod
    def _create_exp_gaus(x, particle, suffix):
        pars = {
            'alpha': RooRealVar('alpha', 'alpha', 1., 0.01, 100.),
            'offset': RooRealVar('offset', 'offset', -5., -20., 0.),
            'mean': RooRealVar('mean_bkg', 'mean_bkg', -5., -10, -3),
            'sigma': RooRealVar('sigma_bkg', 'sigma_bkg', 1.0, 0.1, 5.),
        }
        if particle == 'He3':
            pars['mean'].setVal(-5.)
            pars['mean'].setRange(-10., -2.5)
        exp = RooGenericPdf(f'bkg{suffix}_exp', 'bkg', 
                           f'exp(-alpha * ({x.GetName()} - offset))', 
                           [x, pars['alpha'], pars['offset']])
        gaus = RooGaussian(f'bkg{suffix}_gaus', 'bkg', x, 
                          pars['mean'], pars['sigma'])
        return [exp, gaus], pars
    
    @staticmethod
    def _create_pol1_exp(x, particle, suffix):
        pars = {
            'p0': RooRealVar('p0', 'c_{0}', 1., 0., 1e6),
            'p1': RooRealVar('p1', 'c_{1}', 0., -1e4, 1e4),
            'alpha': RooRealVar('alpha', 'alpha', 1., 0.01, 100.),
            'offset': RooRealVar('offset', 'offset', -5., -20., 0.),
        }
        pol1 = RooGenericPdf(f'bkg{suffix}_pol1', 'bkg', 
                            f'p0 + p1 * {x.GetName()}', 
                            [x, pars['p0'], pars['p1']])
        exp = RooGenericPdf(f'bkg{suffix}_exp', 'bkg', 
                           f'exp(-alpha * ({x.GetName()} - offset))', 
                           [x, pars['alpha'], pars['offset']])
        return [pol1, exp], pars
    
    @staticmethod
    def _create_pol0_gausexp(x, particle, suffix):
        pars = {
            'mean': RooRealVar('bkg_mean', 'mean', -4., -10, -2),
            'sigma': RooRealVar('bkg_sigma', 'sigma', 1.0, 0.1, 5.),
            'rlife': RooRealVar('bkg_rlife', 'rlife', 2., 0.5, 10.),
        }
        pol0 = RooGenericPdf(f'bkg{suffix}_pol0', 'bkg', '1', [x])
        gausexp = RooGausExp(f'bkg{suffix}_gausexp', 'bkg', x, 
                            pars['mean'], pars['sigma'], pars['rlife'])
        return [pol0, gausexp], pars
    