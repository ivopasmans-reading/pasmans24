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
from fig_shared import FigureLines, portrait, nice_ticks
import numpy as np
from dapper.tools.chronos import Chronology
from dapper.mods.Pasmans24 import exp_setup as exp
from dapper.da_methods import E3DVar
import dapper.tools.localization as loc
import matplotlib.pyplot as plt
import os
import dill
from datetime import timedelta, datetime
from scipy.stats import randint
from dapper.mods.Pasmans24.calc_rmse_Nobs_order_noloc import run, io, SubEnsemble
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
        self.xtrue = {}

    def add_order(self, xp, order):
        label = 'DG{:02d}'.format()

    def preplot(self):
        self.create_panels((3, 3), loc=(-.24, 1.0))
        self.fig.set_size_inches(
            portrait['width'], .8*portrait['height'], forward=True)
        self.fig.subplots_adjust(right=.96, left=.13, top=.92, bottom=.15,
                                 wspace=.4)

    def add_Nobs(self, axes, Nobs):
        errors, scales, models = np.array([]), np.array([], dtype=float), np.array([], dtype=str)
        for scale in self.errors:
            for model in self.errors[scale]:
                if model[1] == int(Nobs*exp.Ncells):
                    errors = np.append(errors, self.errors[scale][model])
                    scales = np.append(scales, scale)
                    models = np.append(models, model[0])

        errors = np.reshape(errors, (len(scales), 3, 3))
        isort = np.argsort(scales)
        errors = errors[isort, :, :]
        scales = scales[isort]
        models = models[isort]

        for model in np.append(['Gridpoint'],np.unique(models[models!='Gridpoint'])):
            mask = models == model
            for n, ax in enumerate(axes):
                self.add_band(ax, model, scales[mask],
                              errors[mask, n].T, alpha=.3)

    def add_xp(self, xp):
        opts = dict(node_options(xp))
        scale = opts['_slope']
        if opts['_model_type']=='dg':
            model = ('DG{:02d}'.format(opts['_order']), opts['_Nobs'])
        elif opts['_model_type']=='lin':
            model = ('Gridpoint', opts['_Nobs'])

        if scale not in self.errors:
            self.errors[scale] = {}

        if scale not in self.xtrue:
            self.xtrue[scale] = {}
            self.errors[scale] = {}
            xtrue = xp.xx[1:]

            # truth
            for ndiff in range(3):
                self.xtrue[scale][ndiff] = xp.true_HMM.model.interpolator(
                    xtrue, ndiff=ndiff)(self.r)

        # models
        if model not in self.errors[scale]:

            xfor = np.mean(xp.E['for'][1:], axis=1)
            xana = np.mean(xp.E['ana'], axis=1)
            self.errors[scale][model] = np.zeros((3, 3))

            for ndiff in range(3):
                xtrue1 = self.xtrue[scale][ndiff]
                xfor1 = xp.HMM.model.interpolator(xfor, ndiff=ndiff)(self.r)
                xana1 = xp.HMM.model.interpolator(xana, ndiff=ndiff)(self.r)

                error2 = self.rmse(xana1-xtrue1, axis=1)**2 / self.rmse(xfor1-xtrue1, axis=1)**2
                self.errors[scale][model][ndiff,:] = self.confidence(np.sqrt(error2))

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
        Nobs = [1,3,9]

        for n, No in enumerate(Nobs):
            self.add_Nobs(self.axes[3*n:3*n+3], No)

        for n, ax in enumerate(self.axes[:3]):
            title = r"RMSE $\frac{\mathrm{d}^{"+str(n)+r"} x}{\mathrm{d}r^{"+str(n)+ "}}$"
            ax.set_title(title)
        for ax in self.axes[0::3]:
            lims,ticks = nice_ticks((0,1))
            ax.set_yticks(ticks)
            ax.set_ylim(lims)
        for ax in self.axes[1::3]:
            lims,ticks = nice_ticks((0.95,1.3))
            ax.set_yticks(ticks)
            ax.set_ylim(lims)
        for ax in self.axes[2::3]:
            lims,ticks = nice_ticks((0.95,1.3))
            ax.set_yticks(ticks)
            ax.set_ylim(lims)
            
        for n, ax in enumerate(self.axes[::3]):
            ax.set_ylabel(r'$\frac{\mathrm{RMSE}_{\rm{a}}}{\mathrm{RMSE}_{\rm{b}}}$'
                          + r'($\frac{N_{obs}}{M}=$'+'{:d})'.format(Nobs[n]),
                          **self.label_font)
        for ax in self.axes[6:]:
            ax.set_xlabel(r'$\alpha$', **self.label_font)
        for ax in self.axes:
            ax.grid(visible=True)
            
            lims,ticks = nice_ticks((-4,-1),5)
            ax.set_xticks(ticks)
            ax.set_xlim(lims)
            

        self.add_legend(4,[.2,-.02,.6,.12])


if __name__ == '__main__':
    tree = run()

    truths = exp.collect(tree, 1)
    models = exp.collect(tree, 3)
    xps = exp.collect(tree, 4)

    fig = FigureSlopeError('figT04_pasmans_rmse_slope_order_noloc_sigo10.png')
    sel = select(xps, criteria={'_Nobs': [int(1*exp.Ncells), int(3*exp.Ncells), int(9*exp.Ncells)],
                                '_sig': [1.0]})
    for xp in sel:
        fig.add_xp(xp)
    if len(sel)==0:
        raise ValueError('No experiments in selection.')
        
    fig.preplot()
    fig.plot()
