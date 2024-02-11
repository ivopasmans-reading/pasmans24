#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figures for paper. 

@author: ivo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculations for paper. 
Experiment generating massive ensemble used to study localization and 
covariance errors. 
"""


# Directory

import re
from fig_shared import FigureLines, portrait, nice_ticks, uniform_lim
import numpy as np
from dapper.tools.chronos import Chronology
from dapper.mods.Pasmans24 import exp_setup as exp
from dapper.da_methods import E3DVar
import dapper.tools.localization as loc
import matplotlib.pyplot as plt
import os
from dapper.mods.Pasmans24.calc_rmse_Nobs_order_sigo2 import run, io
from scipy.stats import bootstrap

io.try_loading = True
plt.rcParams.update({'text.usetex':True})


def node_options(node, attrs=[]):
    if node.parent is not None:
        attrs = node_options(node.parent, attrs)
    attrs += [(att, getattr(node, att)) for att in dir(node) if
              re.match('_[^_]', att) is not None]
    return attrs


def select(xps, criteria={}):
    matches = []
    for xp in xps:
        options = dict(node_options(xp))
        if not set(criteria).issubset(set(options)):
            continue
        if any((options[key] not in values for key, values in criteria.items())):
            continue
        matches.append(xp)

    return matches


class FigureSlopeError(FigureLines):

    def __init__(self, fig_name, criteria={}):
        super().__init__()
        self.xscale = 1e6
        self.fig_name = fig_name
        self.r = np.linspace(0, exp.L, 40000, endpoint=False)
        self.errors = {}
        self.criteria = criteria
        self.xtrue = None

    def add_order(self, xp, order):
        label = 'DG{:02d}'.format()

    def preplot(self):
        self.create_panels((3, 3), loc=(-.24, 1.0))
        self.fig.set_size_inches(
            portrait['width'], .8*portrait['height'], forward=True)
        self.fig.subplots_adjust(right=.96, left=.13, top=.92, bottom=.15,
                                 wspace=.4)

    def add_xp(self, xp):
        opts = dict(node_options(xp))
        
        if opts['_model_type']=='lin':
            model = 'nodal'
        else:
            model = 'DG{:02d}'.format(opts['_order'])
        Nobs = opts['_Nobs']
        sigo = opts['_sigo_inflation']
        
        if Nobs not in self.errors:
            self.errors[Nobs] = {}
        if model not in self.errors[Nobs]:
            self.errors[Nobs][model] = {}
        if sigo not in self.errors[Nobs][model]:
            self.errors[Nobs][model][sigo] = {}
            
        if self.xtrue is None:
            xtrue = xp.xx[1:]
            self.xtrue = {}
            for ndiff in range(3):
                self.xtrue[ndiff] = xp.true_HMM.model.interpolator(xtrue, ndiff=ndiff)(self.r)

        xfor = np.mean(xp.E['for'][1:], axis=1)
        xana = np.mean(xp.E['ana'], axis=1)
        self.errors[Nobs][model][sigo] = np.zeros((3, 3))
       
        for ndiff in range(3):
            xtrue1 = self.xtrue[ndiff]
            xfor1 = xp.HMM.model.interpolator(xfor, ndiff=ndiff)(self.r)
            xana1 = xp.HMM.model.interpolator(xana, ndiff=ndiff)(self.r)

            error2 = self.rmse(xana1-xtrue1, axis=1)**2 / self.rmse(xfor1-xtrue1, axis=1)**2
            self.errors[Nobs][model][sigo][ndiff,:] = self.confidence(np.sqrt(error2))

    def confidence(self, errors):
        alpha = 0.9

        vals = np.zeros((3,))
        errors = np.reshape(errors, (1, -1))
        vals[0] = self.rmse(errors, axis=1)
        boot = bootstrap(errors, self.rmse,
                         confidence_level=alpha, method='percentile',
                         n_resamples=100, vectorized=True)
        vals[1] = boot.confidence_interval.low
        vals[2] = boot.confidence_interval.high

        return vals

    def rmse(self, errors, axis=0):
        return np.sqrt(np.mean(errors**2, axis=axis))

    def plot(self):
        
        axes = self.axes.reshape((3,3))
        
        for ax1, key1 in zip(axes, self.errors.keys()):
            Nobs = int(key1/exp.Ncells)
            ax1[0].set_ylabel(r'$\frac{\mathrm{RMSE}_{\rm{a}}}{\mathrm{RMSE}_{\rm{b}}}$'
                              + r'($\frac{N_{obs}}{M}=$'+'{:d})'.format(Nobs),
                              **self.label_font)
            
            for ndiff,ax2 in enumerate(ax1):
                for key2 in self.errors[key1]:
                    sigs = np.array([s for s in self.errors[key1][key2]],dtype=float)
                    vals = np.array([v[ndiff] for v in self.errors[key1][key2].values()])
                    self.add_band(ax2, key2, sigs, vals.T, alpha=.3)
                
        for n, ax in enumerate(self.axes[:3]):
            title = r"RMSE $\frac{\mathrm{d}^{"+str(n)+r"} x}{\mathrm{d}r^{"+str(n)+ "}}$"
            ax.set_title(title)
       
        #ylim
        uniform_lim(self.axes, 'x', max_ticks=5)
        for ax1 in axes.T:
            uniform_lim(ax1, 'y', max_ticks=9)
        
        for ax in self.axes[6:]:
            ax.set_xlabel(r'$\sigma_{obs}$', usetex=True, **self.label_font)
        for ax in self.axes:
            ax.grid(visible=True)
            
        self.add_legend(4,[.2,-.02,.6,.12])


if __name__ == '__main__':
    tree = run()

    truths = exp.collect(tree, 1)
    models = exp.collect(tree, 3)
    xps = exp.collect(tree, 4)

    fig = FigureSlopeError('fig04_pasmans_rmse_slope_order_noloc_sigoR40.png')
    sel = select(xps, criteria={'_Nobs': [int(1*exp.Ncells), int(3*exp.Ncells), int(9*exp.Ncells)],
                                '_slope':[-4.0]})
    for xp in sel:
        fig.add_xp(xp)
    fig.preplot()
    fig.plot()
    fig.save()
