# -*- coding: utf-8 -*-
"""
Plot Legendre polynomials and their fourier transform. 
"""


import numpy as np
import matplotlib as mpl
from scipy.fft import fft, fftfreq
from matplotlib import pyplot as plt
from scipy.special import legendre
from fig_shared import portrait, Figure, FigureLines

plt.rcParams.update({'text.usetex':True})

class FigureLegendre(FigureLines):
    
    fig_name = 'fig05_pasmans_LegendrePolynomials.png'

    def __init__(self):
        super().__init__()
        self.drawings = {}
        self.styles = {}
        self.colors = {}
        
    def preplot(self):
        self.create_panels((1,2),loc=(-.28,.96))
        self.fig.set_size_inches(portrait['width'], 4, forward=True)
        self.fig.subplots_adjust(right=.96, left=.14, top=.96, bottom=.26,
                                 wspace=.42)
        
    def postplot(self):
        self.create_layout_space(self.axes[0])
        self.create_layout_fourier(self.axes[1])
        self.add_legend(3)
        return self.fig, self.axes

    def create_layout_space(self, ax):
        self.default_layout(ax)
        ax.set_xlabel(r'$r$', **self.label_font)
        ax.set_ylabel(r'$\phi_{lm}$', **self.label_font)
        
        ax.plot(np.array([-1,-1]), 1e6*np.array([-1,1]), 'k-', 
                linewidth=1.1*self.linewidth)
        ax.plot(np.array([ 1, 1]), 1e6*np.array([-1,1]), 'k-', 
                linewidth=1.1*self.linewidth)
        ax.plot(np.array([-1e6, 1e6]), np.array([0,0]), 'k-', 
                linewidth=1.1*self.linewidth)
        
        #ax.annotate('cell\n m-1', (-1.5,-.95), xycoords='data', 
        #            **self.label_font, horizontalalignment='center')
        #ax.annotate('cell\n  m ', ( 0.0,-.95), xycoords='data', 
        #            **self.label_font, horizontalalignment='center')
        #ax.annotate('cell\n m+1', ( 1.5,-.95), xycoords='data', 
        #            **self.label_font, horizontalalignment='center')
        
        ax.set_xlim(-2,2)

    def create_layout_fourier(self, ax):
        self.default_layout(ax)
        ax.set_xlabel(r'$\kappa \,[\mathrm{cycles}^{-1}]$', **self.label_font)
        ax.set_ylabel(r'$\mathcal{S} \,[\mathrm{dB}]$', **self.label_font)
        ax.set_ylim(-100,-30)
        ax.set_xscale('log')
        
    def update_ylim(self, ax, y):
        self.ylims[ax][0] = min(self.ylims[ax][0], np.min(y))
        self.ylims[ax][1] = max(self.ylims[ax][1], np.max(y)+1e-8)
        ax.set_ylim(1.05*self.ylims[ax])

    def add_line_space(self, label, x, y):
        ax = self.axes[0]

        if label not in self.colors:
            self.colors[label] = self.color_iterator.__next__()
        if label not in self.styles:
            self.styles[label] = self.style_iterator.__next__()

        drawing, = ax.plot(x, y, label=label, color=self.colors[label],
                          linestyle=self.styles[label], linewidth=self.linewidth)

        if label not in self.drawings:
            self.drawings[label] = drawing
        
        self.update_xlim(ax, x)
        self.update_ylim(ax, y)
        
    def add_line_fourier(self, label, x, y):
        ax = self.axes[1]

        if label not in self.colors:
            self.colors[label] = self.color_iterator.__next__()
        if label not in self.styles:
            self.styles[label] = self.style_iterator.__next__()
          
        fs = (len(x)-1)/(np.max(x)-np.min(x))
        S, f, drawing = ax.magnitude_spectrum(y, Fs=fs, sides='onesided', 
                                              label = label, color=self.colors[label],
                                              linestyle = self.styles[label],
                                              scale='dB', linewidth=1)

        if label not in self.drawings:
            self.drawings[label] = drawing
            
        mask = np.logical_and(f>0, f<3)
        self.update_xlim(ax, f[mask])


def create_fig_legendre(orders):
    fig = FigureLegendre()
    
    ncells = 79

    # space coordinates
    x = np.linspace(-1-ncells, 1+ncells, 256*ncells, endpoint=False)

    #Create canvas
    fig.preplot()
    #Add lines
    for order in orders:
        y = legendre(order)(x)
        y[np.abs(x)>1] *= 0.
        print('mean',np.mean(y))
        fig.add_line_space(rf"order {order}", x, y)
        fig.add_line_fourier(rf"order {order}", x, y)
        
    #Finish plot
    fig.postplot()
    
    #Save
    #fig.save()

create_fig_legendre(np.array([0,1,2,4,6,8]))

