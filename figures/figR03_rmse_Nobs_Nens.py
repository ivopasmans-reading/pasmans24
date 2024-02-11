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
from dapper.mods.Pasmans24 import exp_setup as exp
import matplotlib.pyplot as plt
from dapper.mods.Pasmans24.calc_rmse_Nobs_order_noloc import io as io16
from dapper.mods.Pasmans24.calc_rmse_Nobs_order_noloc24 import io as io24
from dapper.mods.Pasmans24.calc_rmse_Nobs_order_noloc32 import io as io32
from dapper.mods.Pasmans24.calc_rmse_Nobs_order_noloc import run as run16
from dapper.mods.Pasmans24.calc_rmse_Nobs_order_noloc24 import run as run24
from dapper.mods.Pasmans24.calc_rmse_Nobs_order_noloc32 import run as run32
from scipy.stats import bootstrap 
import re

io16.try_loading = True
io24.try_loading = True
io32.try_loading = True
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
        self.add_true(xp)
        
        if hasattr(xp.parent, '_order'):
            order = xp.parent._order
        else:
            order = 0
            
        Nobs = int(xp.parent.parent._Nobs)
        Nens = 'N={:2d}'.format(int(xp._Nens))
        DG = 'DG{:02d}'.format(int(xp.parent._order))
        xfor = np.mean(xp.E['for'][1:],axis=1)
        xana = np.mean(xp.E['ana'],axis=1)
        
        if Nens not in self.errors.keys():
            self.errors[Nens] = {}
        if DG not in self.errors[Nens].keys():
            self.errors[Nens][DG] = {}
        if Nobs not in self.errors[Nens][DG]:
            self.errors[Nens][DG][Nobs] = np.empty((np.size(self.xtrue,0),np.size(self.xtrue,1)))
        
        if np.size(self.xtrue,1)!=np.size(xana,0) or np.size(self.xtrue,1)!=np.size(xfor,0):
            raise IndexError("Dimension mismatch.")
        
        for ndiff in range(np.size(self.xtrue,0)):
            xtrue1 = self.xtrue[ndiff]
            xfor1 = xp.HMM.model.interpolator(xfor, ndiff=ndiff)(self.r)
            xana1 = xp.HMM.model.interpolator(xana, ndiff=ndiff)(self.r)
        
            error2 = self.rmse(xana1-xtrue1, axis=1)**2 #/ self.rmse(xfor1-xtrue1, axis=1)**2
            self.errors[Nens][DG][Nobs][ndiff,:] = np.sqrt(error2)

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
    
    def plot_RmseDof(self, ax, label, Nobs, ndiff, alpha=.3):
        errors = self.errors[Nobs]
        
        dofs = np.sort([int(key) for key in errors.keys()])
        vals = np.empty((3, len(dofs)))
        for n,dof in enumerate(dofs):
            vals[0, n] = self.rmse(errors[dof][ndiff], axis=0)
            boot = bootstrap(np.array([errors[dof][ndiff]]), self.rmse,
                             confidence_level=alpha, method='percentile',
                             n_resamples=100, vectorized=True)
            vals[1, n] = boot.confidence_interval.low
            vals[2, n] = boot.confidence_interval.high
            
        self.add_band(ax, label, dofs, vals, alpha=alpha)
        
    def plot_hue(self, ax, values, label, ndiff):
        alpha = 0.9

        vals = np.empty((3, len(values[label])))
        Nobs = np.empty((len(values[label]),))
        for n, (N, errors) in enumerate(values[label].items()):
            Nobs[n] = int(N) / exp.Ncells
            vals[0, n] = self.rmse(errors[ndiff], axis=0)
            boot = bootstrap(np.array([errors[ndiff]]), self.rmse,
                             confidence_level=alpha, method='percentile',
                             n_resamples=100, vectorized=True)
            vals[1, n] = boot.confidence_interval.low
            vals[2, n] = boot.confidence_interval.high
        
        self.add_band(ax, label, Nobs/exp.Ncells, vals, alpha=.3)
        
    def plot(self):
        self.default_layout(self.axes[0])
        
        for Nens, ax in zip(self.errors.keys(), self.axes):
            for DG in self.errors[Nens]:
                self.plot_hue(ax, self.errors[Nens], DG, 0)
            print('Title ',Nens)
            ax.set_title(Nens, usetex=True)
            
        self.add_legend(2)
            
        ax = self.axes[0]
        ax.set_ylabel(r'$\frac{\mathrm{RMSE}_{\rm{a}}}{\mathrm{RMSE}_{\rm{b}}}$', **self.label_font)
        for n,ax in enumerate(self.axes):
            ax.set_xlabel(r'$\frac{N_{obs}}{M}$', **self.label_font)
            self.update_yticks(ax, min_dy=.01)
            ax.grid(visible=True)
            
        ylim = (0.0,1.0)
        for ax in self.axes:
            ylim = (np.max((ylim[0], ax.get_ylim()[0])), 
                    np.min((ylim[1], ax.get_ylim()[1])))
        ylim, yticks = nice_ticks(ylim)
        print('YLIM ',ylim)
        for ax in self.axes:
            ax.set_yticks(yticks)
            ax.set_ylim(ylim)

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
    fig = FigureDofError('figR03_pasmans_rmse_Nobs_Nens.png')
    
    datas = {16:run16(),24:run24(),32:run32()}
    
    for key in datas.keys():
        datas[key] = {'tree':datas[key]}
        tree = datas[key]['tree']
        datas[key]['truths'] = exp.collect(tree, 1)
        datas[key]['models'] = exp.collect(tree, 3)
        datas[key]['xps'] = exp.collect(tree, 4)
        datas[key]['sel'] = select(fig, datas[key]['xps'], 
                                   criteria={'_slope':[-4.0],'_sig':[1.0],
                                             '_order':[4,10]})

    for Nens, data in datas.items():
        print('Nens ',Nens)
        for ixp,xp in enumerate(data['sel']):
            if hasattr(xp.parent, '_order'):
                print(ixp)
                fig.calculate_error(xp)
            
    fig.plot()
    

        

    
