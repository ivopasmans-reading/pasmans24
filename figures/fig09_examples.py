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

plt.rcParams.update({'text.usetex':True})

io.try_loading = True
ndiff=0

def node_options(node, attrs=[]):
    if node.parent is not None:
        attrs = node_options(node.parent, attrs)
    attrs += [(att,getattr(node,att)) for att in dir(node) if 
              re.match('_[^_]',att) is not None]
    return attrs

def select(xps, criteria={}):
    matches = []
    for xp in xps:
        options = dict(node_options(xp))
        if not set(criteria).issubset(set(options)):
            continue 
        if any((options[key] not in values for key,values in criteria.items())):
            continue 
        matches.append(xp)
        
    return matches

class FigureExamples(FigureLines):

    def __init__(self, fig_name, criteria={}):
        super().__init__()
        self.xscale = 1e6
        self.fig_name = fig_name
        
        ncells = 9
        self.r = np.linspace(0, exp.L/exp.Ncells*ncells, 
                             ncells*100, endpoint=False)
        self.true = None
        self.it = 0

    def preplot(self):
        self.create_panels((3,3), loc=(-.2,.96))
        self.fig.set_size_inches(portrait['width'], portrait['height'], forward=True)
        self.fig.subplots_adjust(right=.96, left=.13, top=.94, bottom=.12,
                                 wspace=.36)

    def add_exp(self, ax, xp):
        Nobs = int(xp.HMM.Obs.M)
        
        #observations in domain
        xobs = xp.HMM.model.obs_coords(self.it+1)
        mask = np.logical_and(xobs>=np.min(self.r),xobs<=np.max(self.r))
        y = xp.yy[self.it][mask]
        xobs = xobs[mask]
        
        xtrue = xp.xx[self.it+1]
        xfor = np.mean(xp.E['for'][self.it+1],axis=0)
        xana = np.mean(xp.E['ana'][self.it],axis=0)
        
        xtrue = xp.true_HMM.model.interpolator(xtrue, ndiff=ndiff)(self.r)[0]
        xfor = xp.HMM.model.interpolator(xfor, ndiff=ndiff)(self.r)[0]
        xana = xp.HMM.model.interpolator(xana, ndiff=ndiff)(self.r)[0]
        
        self.add_line(ax, 'truth', self.r*1e-3, xtrue)
        self.add_line(ax, 'background', self.r*1e-3, xfor)
        self.add_line(ax, 'analysis', self.r*1e-3, xana)
        ax.plot(xobs*1e-3, y, 'k+')
        
    def add_exp_diff(self, ax, xp, alpha):
        Nobs = int(xp.HMM.Obs.M)
        
        xtrue = xp.xx[self.it+1]
        xfor = np.mean(xp.E['for'][self.it+1],axis=0)
        xana = np.mean(xp.E['ana'][self.it],axis=0)
        
        xtrue = xp.true_HMM.model.interpolator(xtrue, ndiff=ndiff)(self.r)[0]
        xfor = xp.HMM.model.interpolator(xfor, ndiff=ndiff)(self.r)[0]
        xana = xp.HMM.model.interpolator(xana, ndiff=ndiff)(self.r)[0]
        
        self.add_line(ax, r'background, $\alpha$', self.r*1e-3, xfor-xtrue)
        self.add_line(ax, r'analysis', self.r*1e-3, xana-xtrue)
        
    def plot(self, xps):
        self.default_layout(self.axes[0])
        general = {'_sig':[1.0],'_Nobs':[int(79*3)]}
        options = [{'_model_type':['lin']},
                  {'_model_type':['dg'],'_order':[2]},
                  {'_model_type':['dg'],'_order':[6]},
                  ]
        axes2d = self.axes.reshape((3,-1))
        
        axes2d[0,0].set_title(r'$S \sim k^{-1}$')
        for ax, opts in zip(axes2d[:,0], options):
            xp = select(xps, {**opts, **general, '_slope':[-1.0]})[0]
            self.add_exp(ax,xp)
    
            if '_order' in opts:
                model_name = '(DG{:02d})'.format(opts['_order'][0])
            else:
                model_name = '(nodal)'
            ax.set_ylabel(r'$x$ {}'.format(model_name),**self.label_font)
            
        axes2d[0,1].set_title(r'$S \sim k^{-4}$')
        for ax, opts in zip(axes2d[:,1], options):
            xp = select(xps, {**opts, **general, '_slope':[-4.0]})[0]
            self.add_exp(ax,xp)
            
        for ax, opts in zip(axes2d[:,2], options):
            xp = select(xps, {**opts, **general, '_slope':[-4.0]})[0]
            self.add_exp_diff(ax,xp,-4.)
        
        for ax in axes2d[-1,:]:
            ax.set_xlabel(r'$r$ [$\mathrm{km}$]', **self.label_font)
        for ax in self.axes:
            self.update_yticks(ax, min_dy=.01)
            ax.grid(visible=True)
        
        self.add_legend(3,[.2,-.06,.6,.12])

if __name__ == '__main__':
    tree = run()

    truths = exp.collect(tree, 1)
    models = exp.collect(tree, 3)
    xps = exp.collect(tree, 4)

    fig = FigureExamples('figR02_pasmans_examples_Nobs.png')
    fig.preplot()
    fig.plot(xps)
    
    

        

    
