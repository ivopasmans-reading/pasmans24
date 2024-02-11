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
import re

io.try_loading = True
plt.rcParams.update({'text.usetex':True})


class FigureNobsError(FigureLines):

    def __init__(self, fig_name, criteria={}):
        super().__init__()
        self.xscale = 1e6
        self.fig_name = fig_name
        self.r = np.linspace(0, exp.L, 40000, endpoint=False)
        self.errors = {}
        self.criteria = criteria
        self.xtrue = None

    #def add_order(self, xp, order):
    #    label = 'DG{:02d}'.format()

    def preplot(self):
        self.create_panels((1,3), loc='title')
        self.fig.set_size_inches(portrait['width'], 5, forward=True)
        self.fig.subplots_adjust(right=.96, left=.13, top=.92, bottom=.25,
                                 wspace=.38)

    def add_true(self, xp):
        xtrue = xp.xx[1:]
        self.xtrue = np.empty((3,np.size(xtrue,0),len(self.r)))
        for ndiff in range(np.size(self.xtrue,0)):
            self.xtrue[ndiff] = xp.true_HMM.model.interpolator(xtrue, ndiff=ndiff)(self.r)

    def add_order(self, xp, label):
        print('Adding ',label)
        Nobs = int(xp.HMM.Obs.M)
        if label not in self.errors:
            self.errors[label] = {}
        if self.xtrue is None:
            self.add_true(xp)
            
        xfor = np.mean(xp.E['for'][1:],axis=1)
        xana = np.mean(xp.E['ana'],axis=1)
        self.errors[label][Nobs] = np.empty((np.size(self.xtrue,0),np.size(self.xtrue,1)))
        
        if np.size(self.xtrue,1)!=np.size(xana,0) or np.size(self.xtrue,1)!=np.size(xfor,0):
            raise IndexError("Dimension mismatch.")
        
        for ndiff in range(np.size(self.xtrue,0)):
            xtrue1 = self.xtrue[ndiff]
            xfor1 = xp.HMM.model.interpolator(xfor, ndiff=ndiff)(self.r)
            xana1 = xp.HMM.model.interpolator(xana, ndiff=ndiff)(self.r)
        
            error2 = self.rmse(xana1-xtrue1, axis=1)**2 / self.rmse(xfor1-xtrue1, axis=1)**2
            self.errors[label][Nobs][ndiff,:] = np.sqrt(error2)
        
    def add_orders(self, xps):         
        for xp in xps:
            options = dict(self.node_options(xp))
            
            for key in self.criteria:
                if key in options and self.criteria[key]!=options[key]:
                    return
            
            if '_model_type' not in options:
                pass 
            elif options['_model_type'] == 'lin':
                self.add_order(xp, 'Gridpoint')
            elif options['_model_type'] == 'dg':
                order = options['_order']
                self.add_order(xp,'DG{:02d}'.format(order))
                
    def node_options(self, node, attrs=[]):
        if node.parent is not None:
            attrs = self.node_options(node.parent, attrs)
        attrs += [(att,getattr(node,att)) for att in dir(node) if 
                  re.match('_[^_]',att) is not None]
        return attrs

    def rmse(self, errors, axis=0):
        return np.sqrt(np.mean(errors**2, axis=axis))

    def plot_hue(self, ax, label, ndiff):
        alpha = 0.9

        vals = np.empty((3, len(self.errors[label])))
        Nobs = np.empty((len(self.errors[label]),))
        for n, (N, errors) in enumerate(self.errors[label].items()):
            Nobs[n] = N
            
            vals[0, n] = self.rmse(errors[ndiff], axis=0)
            boot = bootstrap(np.array([errors[ndiff]]), self.rmse,
                             confidence_level=alpha, method='percentile',
                             n_resamples=100, vectorized=True)
            vals[1, n] = boot.confidence_interval.low
            vals[2, n] = boot.confidence_interval.high
        
        self.add_band(ax, label, Nobs/exp.Ncells, vals, alpha=.3)
        
    def plot(self):
        self.default_layout(self.axes[0])
        
        for key in self.errors.keys():
            self.plot_hue(self.axes[0], key, ndiff=0)
            self.plot_hue(self.axes[1], key, ndiff=1)
            self.plot_hue(self.axes[2], key, ndiff=2)
            
        self.add_legend(4)
        ax = self.axes[0]
        ax.set_ylabel(r'$\frac{\mathrm{RMSE}_{\rm{a}}}{\mathrm{RMSE}_{\rm{b}}}$', **self.label_font)
        for n,ax in enumerate(self.axes):
            title = r"RMSE $\frac{\mathrm{d}^{"+str(n)+r"} x}{\mathrm{d}r^{"+str(n)+ "}}$"
            ax.set_title(title,usetex=True)
            ax.set_xlabel(r'$\frac{N_{obs}}{M}$', **self.label_font)
            self.update_yticks(ax, min_dy=.01)
            ax.grid(visible=True)
        
def plot_DA(xps):
    plt.close('all')
    r = np.linspace(0, exp.L, 2000, endpoint=False)

    x = xps[0].true_HMM.model.interpolator(truths[0].xx[-1])(r)
    plt.plot(r, x[0], 'k-', label='truth')

    xx = np.mean(xps[0].E['for'][-1], axis=0)
    x = xps[0].HMM.model.interpolator(xx)(r)
    plt.plot(r, x[0], 'g-', label='gridpoint for')

    xx = np.mean(xps[0].E['ana'][-1], axis=0)
    x = xps[0].HMM.model.interpolator(xx)(r)
    plt.plot(r, x[0], 'g--', label='gridpoint ana')

    xx = np.mean(xps[1].E['for'][-1], axis=0)
    x = xps[1].HMM.model.interpolator(xx)(r)
    plt.plot(r, x[0], 'r-', label='dg for')

    xx = np.mean(xps[1].E['ana'][-1], axis=0)
    x = xps[1].HMM.model.interpolator(xx)(r)
    plt.plot(r, x[0], 'r--', label='dg ana')

    plt.grid()
    plt.legend()

def select(fig, xps, criteria={}):
    matches = []
    for xp in xps:
        options = dict(fig.node_options(xp))
        if not set(criteria).issubset(set(options)):
            continue 
        if any((options[key] not in values for key,values in criteria.items())):
            continue 
        matches.append(xp)
        
    return matches

if __name__ == '__main__':
    tree = run()

    truths = exp.collect(tree, 1)
    models = exp.collect(tree, 3)
    xps = exp.collect(tree, 4)

    fig = FigureNobsError('figT01_rmse_Nobs_order_slope04_noloc_sigo10.png')
    sel = select(fig, xps, criteria={'_slope':[-4.0],'_sig':[1.0]})
    if len(sel)==0:
        raise ValueError('No experiments in selection.')
    
    fig.add_orders(sel)
    fig.preplot()
    fig.plot()
    

        

    
