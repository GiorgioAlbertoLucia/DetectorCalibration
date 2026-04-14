"""
Shared fitting utilities used by ITS, TPC, and TOF calibrators.

These are direct consolidations of the helpers that previously lived
in utils/utils.py across multiple repos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from ROOT import (
    RooRealVar, RooAbsPdf, RooDataHist,
    RooArgSet, RooFit, TPaveText, TH1, TH1F
)


# ── Slice fitting ─────────────────────────────────────────────────────────────

def calibration_fit_slice(model, hist: TH1F, x: RooRealVar, signal_pars, pt_low_edge, pt_high_edge):
    '''
        Fit a slice of the TOF mass histogram. Return the frame and the fit results

        Parameters
        ----------
        model (RooAbsPdf): model to be fitted
        hist (TH1F): histogram to be fitted
        x (RooRealVar): variable to be fitted
        signal_pars (dict): dictionary with the signal parameters
        pt_low_edge (float): lower edge of the pT bin
        pt_high_edge (float): higher edge of the pT bin

        Returns
        -------
        frame (RooPlot): frame with the fit results
        fit_results (dict): dictionary with the fit results
            - mean (float): mean value
            - mean_err (float): mean error
            - sigma (float): sigma value
            - sigma_err (float): sigma error
            - resolution (float): resolution value
            - resolution_err (float): resolution error
    '''

    datahist = RooDataHist(f'dh_{pt_low_edge:.2f}_{pt_high_edge:.2f}', f'dh_{pt_low_edge:.2f}_{pt_high_edge:.2f}', [x], Import=hist)
    model.fitTo(datahist, PrintLevel=-1)

    frame = x.frame(Title=f'{pt_low_edge:.2f} < p_{{T}} < {pt_high_edge:.2f} GeV/#it{{c}}')
    datahist.plotOn(frame)
    model.plotOn(frame, LineColor=2)
    model.paramOn(frame)
    for icomp, component in enumerate(model.getComponents(), start=3):
        #component.plotOn(frame, LineColor=icomp, LineStyle='--')
        model.plotOn(frame, Components={component}, LineColor=icomp, LineStyle='--')

    mean_err = signal_pars['sigma'].getVal() / np.sqrt(hist.Integral()) if hist.Integral() > 0 else 0.
    resolution = signal_pars['sigma'].getVal() / signal_pars['mean'].getVal()
    resolution_error = resolution * np.sqrt((mean_err / signal_pars['mean'].getVal())**2 + (signal_pars['sigma'].getError() / signal_pars['sigma'].getVal())**2)
    fit_results = {
        'mean': signal_pars['mean'].getVal(),
        'mean_err': mean_err,
        'sigma': signal_pars['sigma'].getVal(),
        'sigma_err': signal_pars['sigma'].getError(),
        'resolution': resolution,
        'resolution_err': resolution_error,
    }

    frame_pull = x.frame(Title=f'')
    hpull = frame.pullHist()
    frame_pull.addPlotable(hpull, 'P')

    return frame, frame_pull, fit_results

# ── Histogram initialisation ──────────────────────────────────────────────────


def initialize_means_and_covariances_kmeans(hist: TH1F, n_components: int):
    '''
        Initialize means and sigmas using KMeans clustering.
        They are ordered from the lowest mean value to the highest.
        hist: histogram to be fitted
        n_components: number of components to fit
    '''

    data_points = []
    for ibin in range(1, hist.GetNbinsX()+1):
        data_points.extend([hist.GetBinCenter(ibin)] * int(hist.GetBinContent(ibin)))

    data_points = np.array(data_points)

    if len(data_points) <= 0:
        print('No data points to fit')
        return

    kmeans = KMeans(n_clusters=n_components, init='k-means++', n_init='auto').fit(data_points.reshape(-1, 1))
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    covariances = []
    for icomp in range(n_components):
        comp_data = data_points[np.where(np.array(labels)==icomp)[0]]
        covariances.append(np.cov(comp_data.T))

    # Sort centers and get the sorted indices
    sorted_indices = np.argsort(centers.flatten())
    #sorted_indices = sorted_indices[::-1]

    # Reorder centers, covariances, and weights based on the sorted indices
    centers = centers[sorted_indices]
    centers = [center[0] for center in centers]
    covariances = [float(covariances[i]) for i in sorted_indices]
    return centers, covariances

def initialize_means_and_covariances_gaussian_mixture(hist: TH1F, n_components: int):
    '''
        Initialize means and sigmas using KMeans clustering.
        They are ordered from the lowest mean value to the highest.
        hist: histogram to be fitted
        n_components: number of components to fit
    '''

    data_points = []
    for ibin in range(1, hist.GetNbinsX()+1):
        data_points.extend([hist.GetBinCenter(ibin)] * int(hist.GetBinContent(ibin)))

    data_points = np.array(data_points).reshape(-1, 1)

    if len(data_points) <= 0:
        print('No data points to fit')
        return

    gaussian_mixture = GaussianMixture(n_components=n_components, covariance_type='full', n_init=10)
    gaussian_mixture.fit(data_points)

    centers = gaussian_mixture.means_.flatten()
    covariances = gaussian_mixture.covariances_.flatten()
    
    sorted_indices = np.argsort(centers)
    centers = centers[sorted_indices]
    covariances = [covariances[i] for i in sorted_indices]
    return centers, covariances

_intialise_means_and_covariances = {
    'kmeans': initialize_means_and_covariances_kmeans,
    'gaussian_mixture': initialize_means_and_covariances_gaussian_mixture
}

def initialize_means_and_covariances(hist: TH1F, n_components: int, method='gaussian_mixture'):
    '''
        Initialize means and covariances using the specified method.
        hist: histogram to be fitted
        n_components: number of components to fit
        method: method to use for initialization ('kmeans' or 'gaussian_mixture')
    '''

    if method not in _intialise_means_and_covariances:
        raise ValueError(f'Unknown method {method}. Available methods are {list(_intialise_means_and_covariances.keys())}')

    return _intialise_means_and_covariances[method](hist, n_components)
