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
import os
import dill
from fig_shared import portrait, Figure, FigureLines
from matplotlib import pyplot as plt

# Load ensemble from harddisk
io.try_loading = True
plt.rcParams.update({'text.usetex': True})

# Transform ensemble to state space.


def legendre2space(xp, r, E):
    interp = xp.HMM.model.interpolator(E)
    return interp(r)


def loc_taper_state(xp, E):
    r = np.linspace(0, exp.L, np.size(E, 1), endpoint=False)

    batcher = loc.LegendreBatcher(r, 0)
    taperer = loc.OptimalTaperer(0, period=exp.L)
    coorder = loc.FunctionCoorder(lambda t: r)
    localizer = loc.Localizer(batcher, taperer, coorder)

    localizer.update_ensemble(E)
    L = localizer.taperer.L

    return L, localizer.taperer


def loc_taper_dg(xp, E):
    r = np.arange(.5, exp.Ncells) * (exp.L/exp.Ncells)
    order = int(np.size(E, 1)/exp.Ncells - 1)

    batcher = loc.LegendreBatcher(r, order)
    taperer = loc.OptimalTaperer(order, period=exp.L)
    coorder = loc.FunctionCoorder(lambda t: r)
    localizer = loc.Localizer(batcher, taperer, coorder)

    localizer.update_ensemble(E)
    L = localizer.taperer.L

    return L, localizer.taperer


def bootstrap_taper(xp, loc_taper, E_iter, alpha=.9):
    dalpha = (1-alpha)*.5
    L_iter = np.array([loc_taper(xp, E)[0] for E in E_iter])
    print('shape L_iter ',np.shape(L_iter))

    mean = np.mean(L_iter, axis=0)
    ubound = np.quantile(L_iter, 1-dalpha, axis=0)
    lbound = np.quantile(L_iter, dalpha, axis=0)

    return np.stack((mean, lbound, ubound), axis=0)


class FigureTaper(FigureLines):

    def __init__(self, fig_name):
        super().__init__()
        self.xscale = 1e6
        self.fig_name = fig_name

    def preplot(self):
        self.create_panels((1, 2), loc=(-.26, .96))
        self.fig.set_size_inches(portrait['width'], 4., forward=True)
        self.fig.subplots_adjust(right=.96, left=.14, top=.9, bottom=.26,
                                 wspace=.33)

    def postplot(self):
        for ax in self.axes:
            self.default_layout(ax)

        ylim = np.array([self.ylims[ax] for ax in self.axes])
        ylim = (np.min(ylim[:, 0]), 1.01)
        self.axes[0].set_ylabel('localisation coefficient')
        self.axes[0].set_title(r'$\mathbf{L}_{00}$')
        self.axes[1].set_title(r'$\mathbf{L}_{01}$')

        for ax in self.axes:
            ax.set_ylim(ylim)
            ax.set_xlabel(r'distance [$10^{6}$ m]')
            ax.set_xlim(0, .5*exp.L/self.xscale)

        self.add_legend(3)

    def add_state(self, y):
        x = np.arange(0, np.size(y, -1)) * (exp.L/np.size(y, -1))
        x /= self.xscale
        label, color, style = 'non-scale', 'k', '-'
        self.colors[label], self.styles[label] = color, style

        for ax in self.axes:
            self.add_band(ax, label, x, y)

    def add_diag(self, y, order):
        x = np.arange(0, np.size(y, -1)) * (exp.L/np.size(y, -1))
        x /= self.xscale
        self.add_band(self.axes[0], "order {:d}".format(order), x, y)

    def add_row0(self, y, order):
        x = np.arange(0, np.size(y, -1)) * (exp.L/np.size(y, -1))
        x /= self.xscale
        self.add_band(self.axes[1], "order {:d}".format(order), x, y)


def plt_figure(xps, N, fig_name):
    Edg = xps[1].E['for'][-1]
    Estate = xps[0].E['for'][-1]

    E_iter = SubEnsemble(Estate, N, 100)
    Lstate = bootstrap_taper(xps[0], loc_taper_state, E_iter)
    print('shape Lstate ',np.shape(Lstate))

    E_iter = SubEnsemble(Edg, N, 100)
    Ldg = bootstrap_taper(xps[1], loc_taper_dg, E_iter)
    print('shape Ldg ', np.shape(Ldg))

    fig = FigureTaper(fig_name+'N{:02d}'.format(N)+'.png')
    fig.preplot()
    fig.add_state(Lstate[:, 0, 0, :])
    for n in np.array([0, 1, 2, 3, 4]):
        fig.add_diag(Ldg[:, n, n, :], n)
        fig.add_row0(Ldg[:, 0, n, :], n)
    fig.postplot()
    
    return fig


if __name__ == '__main__':
    tree = run()
    xps = exp.collect(tree, 4)

    fig96=plt_figure(xps, 96,'fig07_pasmans_taper96')
    #fig16=plt_figure(xps, 16, 'fig08_pasmans_taper16')
