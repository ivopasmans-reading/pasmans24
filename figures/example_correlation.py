#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 11:17:55 2023

90% confidence interval arround correlation

@author: ivo

"""


import numpy as np 
from matplotlib import pyplot as plt 
from matplotlib.cm import get_cmap
from scipy.special import legendre
from scipy.stats import norm

r = np.linspace(1e-3,4,200)

true_corr = lambda r: np.exp(-.5*r**2)
long_corr = lambda r: np.exp(-.5*r**2/1.5**2)

#Correlation 
def corr(cfunc,r,N):
    Cm = cfunc(r)
    Zm = np.arctanh(Cm)
    
    Z95  = Zm + 2 / np.sqrt(N-3)
    C95  = np.tanh(Z95)
    Z05  = Zm - 2 / np.sqrt(N-3)
    C05  = np.tanh(Z05)
      
    return Cm, C95, C05

def plot(cfunc,N,color):
    Cm, C05, C95 = corr(cfunc,r, N)
    
    plt.fill_between(r,C05,C95, color=color, alpha=.1, label='N={:02d}'.format(N))
    
    plt.grid()
    plt.xlim(0,4)
    plt.ylim(-.4,1.)
    plt.ylabel('Correlation')
    plt.xlabel('Distance')
    
    

plt.close('all')
plot(true_corr,16,'r')
plot(true_corr,96,'b')
plt.plot(r, true_corr(r), 'k-',label='truth')
plt.plot([2,2],[-.4,1.],'k--',label='cutoff')
plt.grid()
plt.legend(loc='lower left',framealpha=1)
plt.savefig('/home/ivo/Figures/pasmans2022a/example_correlation1.png', figsize=(7,7),dpi=600)

plt.close('all')
plot(true_corr,96,'r')
plot(long_corr,96,'b')
plt.plot(r, true_corr(r), 'k-',label='short scale')
plt.plot(r, long_corr(r), 'b-',label='long scale')
plt.plot([2,2],[-.4,1.],'k--',label='cutoff')
plt.grid()
plt.legend(loc='lower left',framealpha=1)
plt.savefig('/home/ivo/Figures/pasmans2022a/example_correlation2.png', figsize=(7,7),dpi=600)