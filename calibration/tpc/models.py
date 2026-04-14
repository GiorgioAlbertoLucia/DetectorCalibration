"""
RooFit PDF factories for ITS cluster-size fits.

Migrated verbatim from ITS.py; function signatures are unchanged so
existing call-sites still work.
"""

from ROOT import RooRealVar, RooCrystalBall, RooGaussian
from torchic.roopdf import RooGausExp


def init_signal_roofit(nsigma_tpc: RooRealVar, function: str = 'crystalball'):

    if function == 'crystalball':
        pars = {
            'mean':  RooRealVar('mean',  'mean',  800., 0., 2000.,  ''),
            'sigma': RooRealVar('sigma', 'sigma', 1,    1,  100,    ''),
            'aL':    RooRealVar('aL',    'aL',    0.7,  30.),
            'nL':    RooRealVar('nL',    'nL',    0.3,  30.),
            'aR':    RooRealVar('aR',    'aR',    0.7,  30.),
            'nR':    RooRealVar('nR',    'nR',    0.3,  30.),
        }
        pdf = RooCrystalBall(
            'signal', 'signal', nsigma_tpc, 
            pars['mean'], pars['sigma'],
            pars['aL'], pars['nL'], doubleSided=True)
            #pars['aR'], pars['nR'])

        return pdf, pars
    
    elif function == 'gausexp':
        pars = {
            'mean':  RooRealVar('mean',  'mean',    800., 0., 2000., ''),
            'sigma': RooRealVar('sigma', 'sigma',   8,    5,  100,   ''),
            'rlife': RooRealVar('rlife', 'rlife',   0.,   10.),
        }
        pdf = RooGausExp('signal', 'signal', nsigma_tpc, *pars.values())
        return pdf, pars
    
    elif function == 'gaus':
        pars = {
            'mean':  RooRealVar('mean',  'mean',    800., 0., 2000., ''),
            'sigma': RooRealVar('sigma', 'sigma',   2,    1,  1000,  ''),
        }
        pdf = RooGaussian('signal', 'signal', nsigma_tpc, *pars.values())
        return pdf, pars
    
    else:
        raise ValueError(f'Unknown function: {function}. Supported functions are "crystalball" and "gausexp".')

def init_background_roofit(nsigma_tpc: RooRealVar, function: str = 'gaus'):

    if function == 'gausexp':
        pars = {
            'mean':  RooRealVar('bkg_mean',  'bkg_mean',  100., 0., 2000., ''),
            'sigma': RooRealVar('bkg_sigma', 'bkg_sigma', 10,   5., 100, ''),
            'rlife': RooRealVar('bkg_rlife', 'bkg_rlife', 0.8,  0., 10.),
        }
        pdf = RooGausExp('bkg', 'bkg', nsigma_tpc, *pars.values())
        return pdf, pars

    elif function == 'gaus':
        pars = {
            'mean': RooRealVar('bkg_mean',   'bkg_mean',   10., 0., 2000., ''),
            'sigma': RooRealVar('bkg_sigma', 'bkg_sigma',  1,   1., 100,   ''),
        }
        pdf = RooGaussian('bkg', 'bkg', nsigma_tpc, *pars.values())
        return pdf, pars
    
    else: 
        raise ValueError(f'Unknown function: {function}. Supported functions are "gausexp" and "gaus".')