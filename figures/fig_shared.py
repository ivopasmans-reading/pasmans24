#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared settings for figures. 
"""

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import ticker
import os
from pathlib import Path

#paper size 
portrait = {'width': 6.3, 'height': 9.7}

#landscape 
landspace = {'width': 9.5, 'height': 6.3}

#available linestyles
linestyle_tuples = [
     ('solid',                 (0, (1, 0))),
     ('dotted',                (0, (1, 1))),
     ('densely dashdotdotted',             (2, (10, 3))),
     ('long dash', (0, (3, 1, 1, 1, 1, 1))),  
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosley dashdotted',    (0, (3, 10, 1, 10, 1, 10))),
     ('loosely dashed',        (0, (5, 10))),
     ('long dash with offset', (5, (10, 3))),
     ]

def nice_ticks(lims, max_ticks=12, symmetric=False):
    ticks = np.array([1,2,2.5,5,10])
    
    dx = np.diff(lims) / (max_ticks-1)
    rem = np.mod(np.log10(dx), 1)
    base = np.log10(dx) - rem
    
    rem = np.min(ticks[ticks >= 10**rem])
    dx = rem * 10**base
    
    if symmetric:
        center = np.round(np.mean(lims)/dx)*dx 
        lims = lims - center
    else:
        center = 0. 
        
    ilims = np.round(lims/dx, 2)     
    lims = np.array([np.floor(ilims[0]), np.ceil(ilims[1])], dtype=int)
    ticks = np.arange(lims[0], lims[1]+1) * dx + center 
    
    return ticks[[0,-1]], ticks

def uniform_lim(axes, axdir, max_ticks=12, symmetric=False):
    lims = []
    for ax in axes:
        get_lim = getattr(ax,'get_'+axdir+'lim')
        lims.append(get_lim())
    
    lims = np.array(lims)
    lims[:,0] = np.min(lims[:,0])
    lims[:,1] = np.max(lims[:,1])
    
    lims, ticks = nice_ticks(lims[0], max_ticks, symmetric)
    
    for ax in axes:
        set_lim = getattr(ax,'set_'+axdir+'lim')
        set_ticks = getattr(ax,'set_'+axdir+'ticks')
        set_lim(lims)
        set_ticks(ticks)

class Figure:
    
    def __init__(self):
        self.label_font = {'fontsize': 12, 'fontweight': 'normal'}
        self.tick_font = {'labelsize': 10}
        self.xlims, self.ylims = {}, {}
    
    def create_panels(self, panels, loc='left'):
        plt.close('all')
        self.fig = plt.figure()
        self.axes = np.reshape(self.fig.subplots(*panels), (-1))
        
        if loc == 'left':
            annotate_coord = (.04, .90)
        elif loc == 'right':
            annotate_coord = (.86, .90)
        elif loc == 'title':
            annotate_coord = (-.26, 1.06)
        else:
            annotate_coord = loc
    
        for n, ax in enumerate(self.axes):
            if len(self.axes)>1:
                ann = ax.annotate('('+chr(97+n)+')', annotate_coord, xycoords='axes fraction',
                                  backgroundcolor='w',**self.label_font)
            self.xlims[ax] = np.array([np.inf,-np.inf])
            self.ylims[ax] = np.array([np.inf,-np.inf])
            
    def add_legend(self, ncol, position=[.2,.02,.6,.12]):
        ax = self.fig.add_axes(position)
        ax.axis('off')
        self.legend = ax.legend(handles=self.drawings.values(), framealpha=1, 
                  ncol=ncol, loc='upper center')
        return ax, self.legend
        
    def save(self, fig_dir=None):
        if fig_dir is None:
            fig_dir = Path(__file__).absolute()
            fig_dir = os.path.dirname(fig_dir)
        fig_path = os.path.join(fig_dir, self.fig_name)
        
        self.fig.savefig(fig_path, dpi=400, format='png')
        

class FigureLines(Figure):

    def __init__(self):
        super().__init__()
        self.linewidth = 1.8
        self.styles, self.colors, self.drawings = {}, {}, {}
        self.color_iterator = (plt.cm.tab10(i) for i in range(20))
        self.style_iterator = (s[1] for s in linestyle_tuples)

    def default_layout(self, ax):
        ax.grid()
        ax.xaxis.set_tick_params(**self.tick_font)
        ax.yaxis.set_tick_params(**self.tick_font)
        
    def add_band(self, ax, label, x, y, alpha=.05):
        
        if label not in self.colors:
            self.colors[label] = self.color_iterator.__next__()
        if label not in self.styles:
            self.styles[label] = self.style_iterator.__next__()
        
        drawing, = ax.plot(x, y[0], label=label, color=self.colors[label],
                          linestyle=self.styles[label], linewidth=self.linewidth)

        ax.fill_between(x, y[1], y[2], color=self.colors[label], 
                        alpha=alpha, linewidth=0.0)

        drawing, = ax.plot(x, y[0], label=label, color=self.colors[label],
                          linestyle=self.styles[label], linewidth=self.linewidth)

        if label not in self.drawings:
            self.drawings[label] = drawing
        
        
        self.update_xlim(ax, x)
        self.update_ylim(ax, y)
        
    def add_line(self, ax, label, x, y):
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

    def update_xlim(self, ax, x):
        self.xlims[ax][0] = min(self.xlims[ax][0], np.min(x))
        self.xlims[ax][1] = max(self.xlims[ax][1], np.max(x)+1e-8)
        ax.set_xlim(self.xlims[ax])

    def update_ylim(self, ax, y):
        self.ylims[ax][0] = min(self.ylims[ax][0], np.min(y))
        self.ylims[ax][1] = max(self.ylims[ax][1], np.max(y)+1e-8)
        ax.set_ylim(self.ylims[ax])
    
    def update_yticks(self, ax, symmetric=False, source=None, min_dy=np.inf):        
        if source is None:
            source = ax 
            
        lims = self.ylims[source]
        if np.diff(lims)<min_dy:
            lims = 0.5*min_dy * np.array([-1,1]) + np.mean(lims)
            
        lims, ticks = nice_ticks(lims, symmetric=symmetric)
        ax.set_ylim(lims)
        ax.yaxis.set_major_locator(ticker.FixedLocator(ticks))
        
        

