#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculations for paper. 
Experiment generating massive ensemble used to study localization and 
covariance errors. 
"""

import numpy as np
from dapper.tools.chronos import Chronology
from dapper.mods.Pasmans24 import exp_setup as exp
from dapper.da_methods import E3DVar
import dapper.tools.localization as loc
import matplotlib.pyplot as plt
import os
from datetime import datetime
from scipy.stats import randint

# Directory
fig_dir = os.path.join(exp.fig_dir,'rmse_Nobs_order_Nens')
if not os.path.exists(fig_dir):
    os.mkdir(fig_dir)
    os.mkdir(os.path.join(fig_dir,'tree'))
io = exp.TreeIO(fig_dir, try_loading=True)

# Iterator that creates subensembles.
class SubEnsemble:

    def __init__(self, E, N, size):
        self.E = E
        self.N = N
        self.size = size
        self.seed = 5000

    def __iter__(self):
        self.N0 = 0
        return self

    def __next__(self):
        if self.size <= 0:
            raise StopIteration
        else:
            np.random.seed(self.seed)
            ind = randint.rvs(0, np.size(self.E, 0), size=(self.N,))
            self.size -= 1
            self.seed += 100
            return self.E[ind]

def exp_spectral_Nobs(T=50, sig_obs=1.0):
    tseq = Chronology(dt=1, dto=1, T=T)
    # Background for ncells=79 and poly order 10
    tree = exp.MeanTreeNode(None, tseq, io=io, K=829, sig=10.0) 

    # Truth
    for slope in np.array([-1.]): 
        tree.add_truth(slope=slope, name='slope{:02d}'.format(int(-10*slope)))
    
    # Observation
    #for sig in np.array([0.5,1.,1.25,1.5,2.]):
    for sig in np.array([1.0]):
        Nobs_list = [int(exp.Ncells * nobs) for nobs in [1,3,9] ]
        for Nobs in Nobs_list:
            tree.add_obs(Nobs, sig, obs_type='uniform',
                          name='Sigo{:.1f}_No{:03d}'.format(sig, Nobs))
    
    # Linear model
    def localization(model):
        batcher = loc.LegendreBatcher(model.dyn_coords, 0)
        taperer = loc.OptimalTaperer(0, period=exp.L,tag='Step')
        coorder = loc.FunctionCoorder(model.obs_coords)
        return loc.Localizer(batcher, taperer, coorder)
    
    tree.add_model('lin', exp.Ncells, name="Nodal")
    
    # DG Model
    for order in np.array([0,1,10]):  
        
        #DG model
        def localization(model):
            order = model.order
            batcher = loc.LegendreBatcher(model.dyn_coords, order)
            taperer = loc.LegendreTaperer(order, period=exp.L)
            coorder = loc.FunctionCoorder(model.obs_coords)
            return loc.Localizer(batcher, taperer, coorder)

        tree.add_model('dg', exp.Ncells, order=order,
                        name='DG{:02d}'.format(order))

    loc_rad = np.array([exp.Ncells,1.] + 12*[1]) * (0.5*exp.L/exp.Ncells)
    loc_rad = np.reshape(loc_rad,(-1,1))
    for Nens in np.array([8,16,32,48,64], dtype=int): #32,48,64
        # Experiment
        tree.add_xp(E3DVar, Nens, loc_rad=exp.L/exp.Ncells) #loc_rad=loc_rad, taper='Step')

    return tree

def run():
    t_start = datetime.now()

    tree = exp_spectral_Nobs()
    comm = exp.MpiComm()
    tree.execute(comm)

    t_end = datetime.now()
    if comm.rank == 0:
        print("Clock time", (t_end-t_start))

    return tree

def plot_DA(truths,models,xps):
    plt.close('all')
    r = np.linspace(0, exp.L, 2000, endpoint=False)
    
    x = truths[0].true_HMM.model.interpolator(truths[0].xx[-1])(r)
    plt.plot(r,x[0],'k-',label='truth')
    
    xx = np.mean(xps[0].E['for'][-1],axis=0)
    x = xps[0].HMM.model.interpolator(xx)(r)
    plt.plot(r,x[0],'g-',label='nodal for')
    
    xx = np.mean(xps[0].E['ana'][-1],axis=0)
    x = xps[0].HMM.model.interpolator(xx)(r)
    plt.plot(r,x[0],'g--',label='nodal ana')
    
    xx = np.mean(xps[1].E['for'][-1],axis=0)
    x = xps[1].HMM.model.interpolator(xx)(r)
    plt.plot(r,x[0],'r-',label='dg for')
    
    xx = np.mean(xps[1].E['ana'][-1],axis=0)
    x = xps[1].HMM.model.interpolator(xx)(r)
    plt.plot(r,x[0],'r--',label='dg ana')
    
    plt.grid()
    plt.legend()

if __name__ == '__main__':
    tree = run()

    truths = exp.collect(tree, 1)
    models = exp.collect(tree, 3)
    xps = exp.collect(tree, 4)
    
    #plot_DA(truths, models, xps)
    
    

