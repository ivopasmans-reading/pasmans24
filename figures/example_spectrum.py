#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:24:05 2023

@author: ivo
"""

import numpy as np
from matplotlib import pyplot as plt
from dapper.mods.Pasmans24 import exp_setup as exp
from scipy import fft


def spectrum(slope):
    K = 829
    # spacing
    dx = exp.L/(2*K+1)
    # Wavenumbers
    k = fft.fftfreq(2*K+1, dx) * 2 * np.pi
    k = k[k >= 0]
    # Window
    if slope<0:
        Frho = np.abs(k[1:])**slope
    else:
        Frho = np.exp(-k[1:]**2 * slope**2)
    Frho = np.append([0.], Frho)
    # Create fourier coefficient
    F = np.sqrt(np.abs(Frho))
    # Normalise coefficients
    F = F / np.linalg.norm(F) * 1
    # Spectrum
    return k, F**2

def plot_spectra(ax):
    
    for slope,style in zip(np.array([-1,-4]),[('-','tab:pink'),('--','tab:red')]):
        k, S = spectrum(slope)
        ax.loglog(k[k>0]*1e3, S[k>0], color=style[1], linestyle=style[0], label=r'$\alpha=$'+'{:.1f}'.format(slope),
                  linewidth=3)
        
    ax.grid()    
    ax.legend()
    ax.set_title(r'$\mathrm{Power spectrum}$')
    ax.set_xlabel(r'$\kappa \: [\mathrm{cycle\,km^{-1}}]$')
    

def plot_error(ax):
    
    for slope,style in zip(np.array([-1,-4]),[('-','tab:pink'),('--','tab:red')]):
        k, S = spectrum(slope)
        r = np.linspace(0,exp.L,len(k))
        
        var = []
        for _ in range(0):
            F = np.sqrt(S) * (np.random.normal(size=np.size(S))+1j*np.random.normal(size=np.size(S)))
            s = np.real(np.sum( np.exp(1j*k[None,:]*r[:,None]) * F[None,:], axis=1))
            var.append(np.var(s))
        print('VAR',slope, np.mean(var))    
        
        F = np.sqrt(S) * (np.random.normal(size=np.size(S))+1j*np.random.normal(size=np.size(S)))
        s = np.real(np.sum( np.exp(1j*k[None,:]*r[:,None]) * F[None,:], axis=1))
        plt.plot(r*1e-3, s, color=style[1], linestyle=style[0], linewidth=3, label=r'$\alpha=$'+'{:.1f}'.format(slope))
        
    ax.grid()
    ax.set_title(r'$\mathrm{Perturbation}$')
    ax.set_xlabel(r'$\mathrm{Position}\,\mathrm{[km]}$')
    ax.set_xlim(0, exp.L*1e-3)
    
    ylim = np.max(np.abs(ax.get_ylim()))
    ax.set_ylim(-ylim,ylim)
    
    ax.legend(framealpha=1)

## Create figure 

plt.close('all')
fig,axs = plt.subplots(1,2)
plot_spectra(axs[0])
plot_error(axs[1])
plt.savefig('/home/ivo/Figures/pasmans2022a/example_spectrum.png',dpi=400,figsize=(5,3))


