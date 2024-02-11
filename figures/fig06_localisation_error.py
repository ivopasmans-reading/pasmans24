#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure for paper. 
Figure that shows localisation factor for different settings. 
"""

import numpy as np
from dapper.mods.Pasmans24 import exp_setup as exp
from dapper.mods.Pasmans24.calc_localized_covariance import run, io, SubEnsemble
import dapper.tools.localization as loc
from scipy.stats import bootstrap
from fig_shared import FigureLines, portrait
from matplotlib import pyplot as plt

# Load ensemble from harddisk
io.try_loading = True
plt.rcParams.update({'text.usetex':True})

def legendre2space(xp, r, E):
    interp = xp.HMM.model.interpolator(E)
    return interp(r)


def calc_cov(E, order, localizer=None):
    N = int(np.size(E, 0))
    M = int(np.size(E, 1)/(order+1))
    E = E - np.mean(E, axis=0, keepdims=True)
    E = E / np.sqrt(N-1)

    cov = E.T@E

    if localizer is not None:
        localizer = localizer(exp.L/M, 'x2x', 0, '')
        for i, c in enumerate(cov):
            loc_ind, loc_coef = localizer([i])
            coef = np.zeros_like(c)
            coef[loc_ind] = loc_coef
            cov[i] *= coef

    return cov

def loc_taper_state(xp, E):
    r = np.linspace(0, exp.L, np.size(E, 1), endpoint=False)

    batcher = loc.LegendreBatcher(r, 0)
    taperer = loc.OptimalTaperer(0, period=exp.L)
    coorder = loc.FunctionCoorder(lambda t: r)
    localizer = loc.Localizer(batcher, taperer, coorder)

    localizer.update_ensemble(E)
    L = localizer.taperer.L

    return L


def loc_taper_dg(xp, E):
    r = np.arange(.5, exp.Ncells) * (exp.L/exp.Ncells)
    order = int(np.size(E, 1)/exp.Ncells - 1)

    batcher = loc.LegendreBatcher(r, order)
    taperer = loc.OptimalTaperer(order, period=exp.L)
    coorder = loc.FunctionCoorder(lambda t: r)
    localizer = loc.Localizer(batcher, taperer, coorder)

    localizer.update_ensemble(E)
    L = localizer.taperer.L

    return L


def bootstrap_taper(xp, loc_taper, E_iter, alpha=.9):
    dalpha = (1-alpha)*.5
    L_iter = np.array([loc_taper(xp, E) for E in E_iter])
    mean = np.mean(L_iter, axis=0)
    ubound = np.quantile(L_iter, 1-dalpha, axis=0)
    lbound = np.quantile(L_iter, dalpha, axis=0)

    return np.stack((mean, lbound, ubound), axis=0)


class FigureLocError(FigureLines):

    def __init__(self, order, fig_name):
        super().__init__()
        self.xscale = 1e6
        self.fig_name = fig_name
        self.norms = ['fro', 2]
        self.errors = {}
        self.sizes = {}
        self.order = order
        self.P = None

    def build_true(self, E):
        self.fig_name += "N{:05d}.png".format(np.size(E, 0))
        self.cov_true = calc_cov(E, 0)
        self.norms_true = np.array([np.linalg.norm(self.cov_true, ord=norm)
                                    for norm in self.norms])
        
    def build_true_dg(self, xp, E):
        r = np.linspace(0, exp.L, np.size(E, 1), endpoint=False)
        identity = np.eye(np.size(E, 1))
        self.P = xp.HMM.model.interpolator(identity)(r)
        self.P = self.P.T
        
        self.cov_true_dg = calc_cov(E@self.P.T, 0)
        self.norms_true_dg = np.array([np.linalg.norm(self.cov_true_dg, ord=norm)
                                       for norm in self.norms])
        
        #self.cov_true_dg = self.cov_true 
        #self.norms_true_dg = self.norms_true

    def add_error_noloc(self, label, xp, E):
        localizer = xp.HMM.Obs.localization
        localizer.update_ensemble(E)
        cov = calc_cov(E, 0)

        errors = np.array([np.linalg.norm(cov-self.cov_true, ord=norm)
                          for norm in self.norms])
        errors = errors/self.norms_true
        errors = errors[None, ...]

        if label not in self.errors:
            self.errors[label] = errors
            self.sizes[label] = np.array([np.size(E, 0)])
        else:
            self.sizes[label] = np.append(self.sizes[label], np.size(E, 0))
            self.errors[label] = np.concatenate(
                (self.errors[label], errors), axis=0)

    def add_error_state(self, label, xp, E):
        localizer = xp.HMM.Obs.localization
        localizer.update_ensemble(E)
        cov = calc_cov(E, 0, localizer)

        errors = np.array([np.linalg.norm(cov-self.cov_true, ord=norm)
                          for norm in self.norms])
        errors = errors/self.norms_true

        errors = errors[None, ...]

        if label not in self.errors:
            self.errors[label] = errors
            self.sizes[label] = np.array([np.size(E, 0)])
        else:
            self.sizes[label] = np.append(self.sizes[label], np.size(E, 0))
            self.errors[label] = np.concatenate(
                (self.errors[label], errors), axis=0)

        return cov

    def add_error_dg(self, label, xp, E):
        localizer = xp.HMM.Obs.localization
        localizer.update_ensemble(E)
        cov = calc_cov(E, self.order, localizer)
        #cov = calc_cov(E, self.order)

        cov = self.P @ cov @ self.P.T

        errors = np.array([np.linalg.norm(cov-self.cov_true_dg, ord=norm)
                          for norm in self.norms])
        errors = errors / self.norms_true_dg
        errors = errors[None, ...]

        if label not in self.errors:
            self.sizes[label] = np.array([np.size(E, 0)])
            self.errors[label] = errors
        else:
            self.sizes[label] = np.append(self.sizes[label], np.size(E, 0))
            self.errors[label] = np.concatenate((self.errors[label], errors),
                                                axis=0)

        return cov

    def plot1(self, label, alpha):
        dq = (1-alpha)*0.5

        sizes = np.unique(self.sizes[label])
        vals = np.empty((len(self.norms), 3, len(sizes)), dtype=float)

        for n, N in enumerate(sizes):
            errors = self.errors[label][self.sizes[label] == N]
            vals[:, 0, n] = np.mean(errors, axis=0)
            boot = bootstrap(errors[None, ...], np.mean, vectorized=True,
                             axis=0, confidence_level=alpha, method='percentile',
                             n_resamples=100)
            vals[:, 1, n] = boot.confidence_interval.low
            vals[:, 2, n] = boot.confidence_interval.high

        for n, (norm, ax) in enumerate(zip(self.norms, self.axes)):
            self.add_band(ax, label, sizes, vals[n, :, :], alpha=.3)

    def plot(self):
        for label in self.errors:
            self.plot1(label, 0.9)

    def preplot(self):
        self.create_panels((1, len(self.norms)), loc=(-.14,.94))
        self.fig.set_size_inches(.8*portrait['width'], 4, forward=True)
        self.fig.subplots_adjust(right=.96, left=.10, top=.92, bottom=.24,
                                 wspace=.33)

    def postplot(self):
        for ax in self.axes:
            self.default_layout(ax)

        self.axes[0].set_ylabel('relative error')
        for ax, norm in zip(self.axes, self.norms):
            ax.set_xlabel(r'members', **self.label_font)
            ax.set_title(norm, **self.label_font)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylim(.1, 5)
            ax.set_xlim(4, 1024)

        self.add_legend(3, [.2, .0, .6, .12])


if __name__ == '__main__':
    tree = run()
    xps = exp.collect(tree, 4)
    Nboot = 20

    E = xps[0].E['for'][-1]
    order = int(np.size(E, 1)/exp.Ncells-1)
    fig = FigureLocError(order, "fig06_localisation_error")
    fig.build_true(E)
    
    Edg = xps[1].E['for'][-1]
    fig.build_true_dg(xps[1], Edg)

    for N in 2**np.arange(2,3): #11
        print("N ", N)
        E_iter = SubEnsemble(xps[0].E['for'][-1], N, Nboot)
        for E in E_iter:
            fig.add_error_noloc('no loc.', xps[0], E)

        E_iter = SubEnsemble(xps[0].E['for'][-1], N, Nboot)
        for E in E_iter:
            covstate = fig.add_error_state('non-scale', xps[0], E)

        E_iter = SubEnsemble(xps[1].E['for'][-1], N, Nboot)
        for E in E_iter:
            covdg = fig.add_error_dg('order 4', xps[1], E)

    fig.preplot()
    fig.plot()
    fig.postplot()
