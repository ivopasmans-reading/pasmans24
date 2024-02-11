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
from dapper.mods.Pasmans24.calc_rmse_Nobs_order_test import run, io, SubEnsemble
from scipy.stats import bootstrap 
import re

io.try_loading = True
plt.rcParams.update({'text.usetex':True})


class FigureNobsError(FigureLines):

    def __init__(self, fig_name, criteria={}):
        super().__init__()
        self.xscale = 1e6
        self.fig_name = fig_name
        
        dr = exp.L/exp.Ncells/40
        self.r = np.arange(80*exp.Ncells) +.5
        self.r = exp.L * self.r / ( 80*exp.Ncells )
        
        
        self.errors = {}
        self.criteria = criteria
        self.xtrue = None

    def add_order(self, xp, order):
        label = 'DG{:02d}'.format()

    def preplot(self):
        plt.figure()
        self.create_panels((3,1), loc='title')
        self.fig.set_size_inches(portrait['width'], 6, forward=True)
        self.fig.subplots_adjust(right=.96, left=.15, top=.92, bottom=.25,
                                 wspace=.38,hspace=.5)

    def add_true(self, xp):
        self.xtrue = np.empty((3,np.size(xp.xx[1:3],0),len(self.r)))
        self.xmean = np.empty((np.size(self.xtrue,1),exp.Ncells))
        for ndiff in range(np.size(self.xtrue,0)):
            for t in np.arange(np.size(self.xtrue,1)):
                print('TIME ',ndiff,t)
                self.xtrue[ndiff,t]  = xp.HMM.model.functions[0](self.r,t,ndiff=ndiff)
                self.xtrue[ndiff,t] += xp.HMM.noise.functions[0](self.r,t,ndiff=ndiff)
                
            if ndiff==0:
                self.xmean[t] = np.mean(self.xtrue[ndiff,t].reshape((exp.Ncells,-1)), axis=1)
            
    def add_orders(self, xps):        
        plt.figure()
        
        for xp in xps:
            options = dict(self.node_options(xp))
            
            for key in self.criteria:
                if key in options and self.criteria[key]!=options[key]:
                    return
            
            if '_model_type' not in options:
                pass 
            elif options['_model_type'] == 'lin':
                self.add_order(xp, 'Nodal')
            elif options['_model_type'] == 'dg':
                order = options['_order']
                self.add_order(xp,'DG{:02d}'.format(order))

    def add_order(self, xp, label):
        print('Adding ',label)
        Nobs = int(xp.HMM.Obs.M)
        if label not in self.errors:
            self.errors[label] = {}
        if self.xtrue is None:
            self.add_true(xp)
        
        xfor = xp.E['for'][1:3,0]
        self.errors[label] = np.empty((3, np.size(xfor,0)))
        
        r = np.linspace(0, exp.L, np.size(xp.xx,1), endpoint=False)
        for ndiff in range(3):
            xfor1  = xp.HMM.model.interpolator(xfor, ndiff=ndiff)(self.r)            
            error2 = self.rmse(xfor1-self.xtrue[ndiff], axis=1)**2
            self.errors[label][ndiff] = np.sqrt(error2)
            
            if ndiff==0:
                plt.plot(self.r, self.xtrue[ndiff,0],'k-')
                plt.plot(self.r, xfor1[0],'--')
        
    def confidence(self, y):
        alpha = 0.9

        vals = np.empty((3,))
        vals[0] = self.rmse(y, axis=0)
        boot = bootstrap(np.array([y]), self.rmse,
                         confidence_level=alpha, method='percentile',
                         n_resamples=100, vectorized=True)
        vals[1] = boot.confidence_interval.low
        vals[2] = boot.confidence_interval.high
        
        return vals
                
    def node_options(self, node, attrs=[]):
        if node.parent is not None:
            attrs = self.node_options(node.parent, attrs)
        attrs += [(att,getattr(node,att)) for att in dir(node) if 
                  re.match('_[^_]',att) is not None]
        return attrs

    def rmse(self, errors, axis=0):
        return np.sqrt(np.mean(errors**2, axis=axis))
        
    def plot(self):
        self.default_layout(self.axes[0])
        
        for ndiff in range(3):
            x = np.arange(len(self.errors.keys()))
            ticks = [key for key in self.errors.keys()]
            y = np.array([self.confidence(self.errors[key][ndiff,:]) for key in 
                 self.errors.keys()])
            
            print('Y',y.T[0])
            self.add_band(self.axes[ndiff], "RMSE_{\rm{b}}", x, y.T)
            self.axes[ndiff].set_xticks(x)
        
        ax = self.axes[0]
        ax.set_ylabel(r'$\mathrm{RMSE}_{\rm{b}}$', **self.label_font)
        for n,ax in enumerate(self.axes):
            title = r"RMSE $\frac{\mathrm{d}^{"+str(n)+r"} x}{\mathrm{d}r^{"+str(n)+ "}}$"
            ax.set_title(title,usetex=True)
            ax.grid(visible=True)
            ax.set_yscale('log')
            ax.set_xticklabels([key for key in self.errors.keys()])
            ax.grid('on')
            
def plot_DA(xps):
    plt.close('all')
    r = np.linspace(0, exp.L, 2000, endpoint=False)

    x = xps[0].true_HMM.model.interpolator(truths[0].xx[-1])(r)
    plt.plot(r, x[0], 'k-', label='truth')

    xx = np.mean(xps[0].E['for'][-1], axis=0)
    x = xps[0].HMM.model.interpolator(xx)(r)
    plt.plot(r, x[0], 'g-', label='nodal for')

    xx = np.mean(xps[0].E['ana'][-1], axis=0)
    x = xps[0].HMM.model.interpolator(xx)(r)
    plt.plot(r, x[0], 'g--', label='nodal ana')

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

    fig = FigureNobsError('figR07_pasmans_rmse_order_slope01.png')
    sel = select(fig, xps, criteria={'_slope':[-1.0],'_sig':[1.0],'_Nobs':[79]})
    fig.add_orders(sel)
    fig.preplot()
    fig.plot()
    

        

    
