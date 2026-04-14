"""
RooFit PDF factories for ITS cluster-size fits.

Migrated verbatim from ITS.py; function signatures are unchanged so
existing call-sites still work.
"""

from ROOT import RooRealVar, RooCrystalBall, RooGaussian, RooGenericPdf, RooAddPdf
from torchic.roopdf import RooGausExp


def init_signal_roofit(x: RooRealVar, function: str = 'gausexp'):
    """
    Create a signal PDF for the ITS cluster-size fit.

    Parameters
    ----------
    x : RooRealVar
        Observable (average cluster size × cos λ).
    function : str
        'crystalball' | 'gausexp' | 'gaus'

    Returns
    -------
    (pdf, pars_dict)
    """
    if function == 'crystalball':
        pars = {
            'mean':  RooRealVar('mean',  'mean',  0.,  5,  ''),
            'sigma': RooRealVar('sigma', 'sigma', 0.01, 1, ''),
            'aL':    RooRealVar('aL',    'aL',    0.7, 30.),
            'nL':    RooRealVar('nL',    'nL',    0.3, 30.),
            'aR':    RooRealVar('aR',    'aR',    0.7, 30.),
            'nR':    RooRealVar('nR',    'nR',    0.3, 30.),
        }
        pdf = RooCrystalBall(
            'signal', 'signal', x,
            pars['mean'], pars['sigma'],
            pars['aL'], pars['nL'], doubleSided=True,
        )
        return pdf, pars

    elif function == 'gausexp':
        pars = {
            'mean':  RooRealVar('mean',  'mean',  0.,  5,  ''),
            'sigma': RooRealVar('sigma', 'sigma', 0.01, 1, ''),
            'rlife': RooRealVar('rlife', 'rlife', 2.,  0., 10.),
        }
        pdf = RooGausExp('signal', 'signal', x, *pars.values())
        return pdf, pars

    elif function == 'gaus':
        pars = {
            'mean':  RooRealVar('mean',  'mean',  1.,  15, ''),
            'sigma': RooRealVar('sigma', 'sigma', 0.01, 1, ''),
        }
        pdf = RooGaussian('signal', 'signal', x, *pars.values())
        return pdf, pars

    else:
        raise ValueError(
            f'Unknown signal function: {function!r}. '
            'Supported: "crystalball", "gausexp", "gaus".'
        )


def init_background_roofit(x: RooRealVar, particle: str, function: str = 'gausexp'):
    """
    Create a background PDF for the ITS cluster-size fit.

    Parameters
    ----------
    x : RooRealVar
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
        pdf = RooGausExp('bkg', 'bkg', x, *pars.values())
        return pdf, pars

    elif function == 'gaus':
        pars = {
            'mean':  RooRealVar('bkg_mean',  'bkg_mean',  0., 1., ''),
            'sigma': RooRealVar('bkg_sigma', 'bkg_sigma', 0.1, 0.8, ''),
        }
        if particle == 'He':
            pars['mean'] = RooRealVar('bkg_mean', 'bkg_mean', 0., 3., '')
        pdf = RooGaussian('bkg', 'bkg', x, pars['mean'], pars['sigma'])
        return pdf, pars
    
    elif function == 'exp+exp':
        pars = {
            'alpha0': RooRealVar('alpha0', 'alpha0', 0.1, 10., ''),
            'offset0': RooRealVar('offset0', 'offset0', 0., 3., ''),
            'alpha1': RooRealVar('alpha1', 'alpha1', -10, -0.1, ''),
            'offset1': RooRealVar('offset1', 'offset1', 0., 3., ''),
            'frac': RooRealVar('frac', 'frac', 0., 1., ''),
        }
        pdf1 = RooGenericPdf('pdf1', 'pdf1', 'exp(-alpha0*(x-offset0))', [x, pars['alpha0'], pars['offset0']])
        pdf2 = RooGenericPdf('pdf2', 'pdf2', 'exp(-alpha1*(x-offset1))', [x, pars['alpha1'], pars['offset1']])
        pdf = RooAddPdf('bkg', 'bkg', [pdf1, pdf2], pars['frac'])
        return pdf, pars
        

    else:
        raise ValueError(
            f'Unknown background function: {function!r}. '
            'Supported: "gausexp", "gaus".'
        )