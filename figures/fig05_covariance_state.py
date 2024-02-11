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
from fig_shared import Figure, FigureLines, portrait
from matplotlib import pyplot as plt
import matplotlib as mpl

# Load ensemble from harddisk
io.try_loading = True
plt.rcParams.update({'text.usetex':True})


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


class FigureCovarianceDiff(Figure):

    def __init__(self, fig_name):
        super().__init__()
        self.xscale = 1e6
        self.fig_name = fig_name
        self.order = 4

    def preplot(self):
        self.create_panels((2,2), loc=(-.22,.94))
        self.fig.set_size_inches((.92*portrait['width'],.62*portrait['height']),
                                 forward=True)
        self.fig.subplots_adjust(right=.99, left=.08, top=.99, bottom=.16,
                                 wspace=.16,hspace=.22)
        
    def color_setting(self, cmax=1.1):
        # Color ticks
        cticks = np.arange(-cmax, 1.001*cmax, .05)
        vlims = (np.min(cticks), np.max(cticks))

        # colorbar with whites in the center
        cm = plt.cm.seismic
        color = [cm(i) for i in range(cm.N)]
        colorvals = np.linspace(vlims[0], vlims[1], cm.N, endpoint=False)
        for i in range(cm.N):
            if np.abs(colorvals[i]) < .05:
                color[i] = (1., 1., 1., 1.)

        # Use ticks to setup color norm
        norm = mpl.colors.BoundaryNorm(cticks, cm.N)
        color = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', color, cm.N)

        return color, vlims, norm
        
    def layout_state_cov(self, ax):
        ax.set_xlabel(r'$r$ [$10^6$ m]', **self.label_font)
        ax.set_ylabel(r'$r$'+"'"+' [$10^6$ m]', **self.label_font)
        ax.set_ylim(0,exp.L/self.xscale)
        ax.set_xlim(0,exp.L/self.xscale)
        ax.set_aspect(1)
        
    def layout_dg_cov(self, ax):
        ax.set_xlabel(r'order', **self.label_font)
        ax.set_ylabel(r'order', **self.label_font)
        ax.set_ylim(0,self.order+1)
        ax.set_xlim(0,self.order+1)
        
        for n in range(self.order+1):
            ax.plot([0,self.order+1],[n,n],'k-')
            ax.plot([n,n],[0,self.order+1],'k-')
        
        labels = ["{:d}".format(n) for n in np.arange(0,self.order+1)]
        ax.set_xticks(np.arange(.5,self.order+1))
        ax.set_yticks(np.arange(.5,self.order+1))
        ax.set_xticklabels(labels, **self.label_font)
        ax.set_yticklabels(labels, **self.label_font)
        ax.set_aspect(1)
        
        
    def add_cov_state(self, ax, cov):
        r=np.linspace(0,exp.L/self.xscale,np.size(cov,0),endpoint=False)
    
        r0, r1 = np.meshgrid(r, r)
        color, vlims, norm = self.color_setting()
        self.drawing = ax.pcolormesh(r0, r1, cov, cmap=color, vmin=vlims[0], 
                                vmax=vlims[1], norm=norm)

    
    def add_cov_dg(self, ax, cov):
        O = self.order+1
        M = int(np.size(cov,0)/O)
        
        cov = np.reshape(cov,(M,O,M,O))
        cov = np.transpose(cov, (1,0,3,2))
        cov = np.reshape(cov,(M*O,M*O))
        
        orders = np.reshape(np.arange(0,self.order+1),(-1,1))
        r = np.reshape(np.arange(0.5, M)/M,(1,-1))
        r = orders+r
    
        r0, r1 = np.meshgrid(r, r)
        color, vlims, norm = self.color_setting()
        self.drawing = ax.pcolormesh(r0, r1, cov, cmap=color, vmin=vlims[0], 
                                     vmax=vlims[1], norm=norm)
        

    def add_colorbar(self):
        color, vlims, norm = self.color_setting()
        cax = self.fig.add_axes((0.14, .05, .8, .02))
        cticks = np.arange(-1.0, 1.01, .2)
        
        cb = plt.colorbar(self.drawing, cax=cax, orientation='horizontal',
                          norm=norm, ticks=cticks, spacing='proportional',
                          extend='both')
        #cb.set_label('Covariance')

    def postplot(self):
        for ax in self.axes[0::2]:
            self.layout_state_cov(ax)
        for ax in self.axes[1::2]:
            self.layout_dg_cov(ax)
        self.add_colorbar()


def calc_cov(E, order, localizer=None):
    N = int(np.size(E,0))
    M = int(np.size(E,1)/(order+1))
    E = E - np.mean(E, axis=0, keepdims=True)
    E = E / np.sqrt(N-1)
    
    cov = E.T@E
    
    if localizer is not None:
        localizer = localizer(exp.L/M,'x2x',0,'')
        for i,c in enumerate(cov):
            loc_ind, loc_coef = localizer([i])
            coef = np.zeros_like(c)
            coef[loc_ind] = loc_coef
            cov[i] *= coef
    
    return cov

def dg2state(xp_dg, order):
    M = int(exp.Ncells * (order+1))
    r = np.linspace(0,exp.L,M,endpoint=False)
    
    identity = np.eye(M)
    P=xp_dg.HMM.model.interpolator(identity)(r)
    return P.T
    

def plt_figure(xps, N, fig_name):
    order = 4
    
    fig = FigureCovarianceDiff(fig_name+"N{:02d}".format(N)+'.png')
    fig.preplot()
    
    #dg2state matrix
    P=dg2state(xps[1], 4)
    Pinv=np.linalg.inv(P)
    
    #state loc.
    E = xps[0].E['for'][-1]
    xps[0].HMM.Obs.localization.update_ensemble(E[:N])
    localizer = xps[0].HMM.Obs.localization
    cov = calc_cov(E, 0, localizer)
    print('cov',np.min(cov),np.max(cov))
    fig.add_cov_state(fig.axes[0], cov)
    
    #state convert
    cov = Pinv @ cov @ Pinv.T
    print('cov',np.min(cov),np.max(cov))
    fig.add_cov_dg(fig.axes[1], cov)
    
    #dg lo.
    E = xps[1].E['for'][-1]
    xps[1].HMM.Obs.localization.update_ensemble(E[:N])
    localizer = xps[1].HMM.Obs.localization
    cov = calc_cov(E[:N], order, localizer)
    print('cov',np.min(cov),np.max(cov))
    fig.add_cov_dg(fig.axes[3], cov)    
    
    #dg convert
    cov = P @ cov @ P.T
    print('cov',np.min(cov),np.max(cov))
    fig.add_cov_state(fig.axes[2], cov)
    
    fig.postplot()
    fig.save()
    
    return P, fig
    

if __name__ == '__main__':
    
    tree = run()
    xps = exp.collect(tree, 4)
    cov, fig =plt_figure(xps, 16, 'fig10_pasmans_CovarianceDiff')
