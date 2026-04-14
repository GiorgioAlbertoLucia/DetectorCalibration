"""
RooFit PDF factories for ITS cluster-size fits.

Migrated verbatim from ITS.py; function signatures are unchanged so
existing call-sites still work.
"""

from ROOT import RooRealVar, RooCrystalBall, RooGaussian
from torchic.roopdf import RooGausExp


def init_signal_roofit(clsize: RooRealVar, function: str = 'gausexp'):
    """
    Create a signal PDF for the ITS cluster-size fit.

    Parameters
    ----------
    clsize : RooRealVar
        Observable (average cluster size × cos λ).
    function : str
        'crystalball' | 'gausexp' | 'gaus'

    Returns
    -------
    (pdf, pars_dict)
    """
    if function == 'dscrystalball':
        pars = {
            'mean':  RooRealVar('mean',  'mean',  1.,  15,  ''),
            'sigma': RooRealVar('sigma', 'sigma', 0.1, 10, ''),
            'aL':    RooRealVar('aL',    'aL',    0.7, 30.),
            'nL':    RooRealVar('nL',    'nL',    0.3, 30.),
            'aR':    RooRealVar('aR',    'aR',    0.7, 30.),
            'nR':    RooRealVar('nR',    'nR',    0.3, 30.),
        }
        pdf = RooCrystalBall(
            'signal', 'signal', clsize,
            pars['mean'], pars['sigma'],
            pars['aL'], pars['nL'], doubleSided=True,
        )
        return pdf, pars
    
    elif function == 'crystalball':
        pars = {
            'mean':  RooRealVar('mean',  'mean',  1.,  15,  ''),
            'sigma': RooRealVar('sigma', 'sigma', 0.1, 10, ''),
            'aL':    RooRealVar('aL',    'aL',    0.7, 30.),
            'nL':    RooRealVar('nL',    'nL',    0.3, 30.),
            'aR':    RooRealVar('aR',    'aR',    0.7, 30.),
            'nR':    RooRealVar('nR',    'nR',    0.3, 30.),
        }
        pdf = RooCrystalBall(
            'signal', 'signal', clsize, 
            pars['mean'], pars['sigma'],
            pars['aL'], pars['nL'],
            pars['aR'], pars['nR'],
        )
        return pdf, pars

    elif function == 'gausexp':
        pars = {
            'mean':  RooRealVar('mean',  'mean',  1.,  15,  ''),
            'sigma': RooRealVar('sigma', 'sigma', 0.01, 10, ''),
            'rlife': RooRealVar('rlife', 'rlife', 2.,  0.9, 10.),
        }
        pdf = RooGausExp('signal', 'signal', clsize, *pars.values())
        return pdf, pars

    elif function == 'gaus':
        pars = {
            'mean':  RooRealVar('mean',  'mean',  1.,  15, ''),
            'sigma': RooRealVar('sigma', 'sigma', 0.01, 10, ''),
        }
        pdf = RooGaussian('signal', 'signal', clsize, *pars.values())
        return pdf, pars

    else:
        raise ValueError(
            f'Unknown signal function: {function!r}. '
            'Supported: "crystalball", "gausexp", "gaus".'
        )


def init_background_roofit(clsize: RooRealVar, particle: str, function: str = 'gausexp'):
    """
    Create a background PDF for the ITS cluster-size fit.

    Parameters
    ----------
    clsize : RooRealVar
        Observable.
    particle : str
        'Pr' or 'He' (affects parameter ranges for He).
    function : str
        'gausexp' | 'gaus'

    Returns
    -------
    (pdf, pars_dict)
    """
    if function == 'gausexp':
        pars = {
            'mean':  RooRealVar('bkg_mean',  'bkg_mean',  0., 3.5, ''),
            'sigma': RooRealVar('bkg_sigma', 'bkg_sigma', 0.1, 0.8, ''),
            'rlife': RooRealVar('bkg_rlife', 'rlife',     2., 0., 10.),
        }
        if particle == 'He':
            pars['mean'] = RooRealVar('bkg_mean', 'bkg_mean', 0., 3., '')
        pdf = RooGausExp('bkg', 'bkg', clsize, *pars.values())
        return pdf, pars

    elif function == 'gaus':
        pars = {
            'mean':  RooRealVar('bkg_mean',  'bkg_mean',  0., 1., ''),
            'sigma': RooRealVar('bkg_sigma', 'bkg_sigma', 0.1, 0.8, ''),
        }
        if particle == 'He':
            pars['mean'] = RooRealVar('bkg_mean', 'bkg_mean', 0., 3., '')
        pdf = RooGaussian('bkg', 'bkg', clsize, pars['mean'], pars['sigma'])
        return pdf, pars

    else:
        raise ValueError(
            f'Unknown background function: {function!r}. '
            'Supported: "gausexp", "gaus".'
        )