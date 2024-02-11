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
from fig_shared import FigureLines, portrait
import numpy as np
from dapper.mods.Liu2002 import exp_setup as exp
import matplotlib.pyplot as plt
from dapper.mods.Pasmans24.calc_rmse_Nobs_resolution import run as run
from dapper.mods.Pasmans24.calc_rmse_Nobs_resolution import io as io
from scipy.stats import bootstrap 
import re

io.try_loading = True
plt.rcParams.update({'text.usetex':True})


class FigureDofError(FigureLines):

    def __init__(self, fig_name, criteria={}):
        super().__init__()
        self.xscale = 1e6
        self.fig_name = fig_name
        self.r = np.linspace(0, exp.L, 40000, endpoint=False)
        self.errors = {}
        self.criteria = criteria
        self.xtrue = None
        self.preplot()

    def preplot(self):
        self.create_panels((1,3), loc='title')
        self.fig.set_size_inches(portrait['width'], 5, forward=True)
        self.fig.subplots_adjust(right=.96, left=.13, top=.92, bottom=.25,
                                 wspace=.38)
        
    def calculate_error(self, xp):
        if self.xtrue is None:
            self.add_true(xp)
            
        Nobs = int(xp.parent.parent._Nobs)
        if Nobs not in self.errors:
            self.errors[Nobs] = {}
        
        if hasattr(xp.parent, '_order'):
            order = xp.parent._order
            model = 'DG'
        else:
            order = 0
            model = 'lin'
        dof = int((order+1) * xp.parent._Ncells)
        
        if model not in self.errors[Nobs]:
            self.errors[Nobs][model] = {}
            
        if dof not in self.errors[Nobs][model]:
            self.errors[Nobs][model][dof] = np.empty((3,1))
        
        xfor = np.mean(xp.E['for'][1:], axis=1)
        xana = np.mean(xp.E['ana'], axis=1)
        for ndiff in range(np.size(self.xtrue,0)):
            xtrue1 = self.xtrue[ndiff]
            print(xp.HMM.model)
            xfor1 = xp.HMM.model.interpolator(xfor, ndiff=ndiff)(self.r)
            xana1 = xp.HMM.model.interpolator(xana, ndiff=ndiff)(self.r)
            
            error2 = self.rmse(xana1-xtrue1, axis=1)**2 / self.rmse(xfor1-xtrue1, axis=1)**2
            self.errors[Nobs][model][dof][:,0] = self.confidence(np.sqrt(error2))

    def add_true(self, xp):
        xtrue = xp.xx[1:]
        self.xtrue = np.empty((3,np.size(xtrue,0),len(self.r)))
        for ndiff in range(np.size(self.xtrue,0)):
            self.xtrue[ndiff] = xp.true_HMM.model.interpolator(xtrue, ndiff=ndiff)(self.r)
                
    def node_options(self, node, attrs=[]):
        if node.parent is not None:
            attrs = self.node_options(node.parent, attrs)
        attrs += [(att,getattr(node,att)) for att in dir(node) if 
                  re.match('_[^_]',att) is not None]
        return attrs

    def rmse(self, errors, axis=0):
        return np.sqrt(np.mean(errors**2, axis=axis))
    
    def confidence(self, errors):
        vals = np.empty((3,))
        alpha = .9
        vals[0] = self.rmse(errors, axis=0)
        boot = bootstrap(np.array([errors]), self.rmse,
                         confidence_level=alpha, method='percentile',
                         n_resamples=100, vectorized=True)
        vals[1] = boot.confidence_interval.low
        vals[2] = boot.confidence_interval.high
        return vals
    
    def plot_RmseDof(self, ax, Nobs, ndiff, alpha=.3):
        errors = self.errors[Nobs]

        for model in errors.keys():
            dofs = np.sort([int(key) for key in errors[model].keys()])
            vals = np.empty((3, len(dofs)))
            for n,dof in enumerate(dofs):
                vals[:,n] = errors[model][dof][:,0]

            self.add_band(ax, model, dofs, vals, alpha=.3)
        
    def plot(self):
        self.default_layout(self.axes[0])
        
        obs_list = np.array([int(n) for n in self.errors.keys()])
        obs_list = np.sort(obs_list) 
        for ax, Nobs in zip(self.axes, obs_list):
            self.plot_RmseDof(ax, Nobs, 0)
            title = r"$N_{obs}$ = " + "{:d}".format(Nobs)
            ax.set_title(title,usetex=True)
            
        ax = self.axes[0]
        ax.set_ylabel(r'$\frac{\mathrm{RMSE}_{\rm{a}}}{\mathrm{RMSE}_{\rm{b}}}$', 
                      **self.label_font)
        for n,ax in enumerate(self.axes):
            ax.set_xlabel(r'degrees of freedom', **self.label_font)
            self.update_yticks(ax, min_dy=.01)
            ax.grid(visible=True)

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

    fig = FigureDofError('figT01_pasmans_rmse_Nobs_Ncells_slope01.png')
    sel = select(fig, xps, criteria={'_slope':[-4.0],'_sig':[1.0]})
    for xp in sel:
        fig.calculate_error(xp)
    
    fig.preplot()
    fig.plot()
    

        

    
