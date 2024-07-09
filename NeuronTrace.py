# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 14:16:52 2015

@author: csnyder
"""
from pyface.qt import QtGui, QtCore

import matplotlib
# We want matplotlib to use a QT backend
matplotlib.use('Qt4Agg')
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from traits.api import Any, Instance
from traitsui.qt4.editor import Editor
from traitsui.qt4.basic_editor_factory import BasicEditorFactory
from traitsui_mpl_qt import MPLFigureEditor


import numpy as np
from traitsui.api import View,Item
from traits.api import Float,Int,HasTraits


class TracePlot(HasTraits):
    figure = Instance(Figure, ())
    T=Int(200)
    y=None
    vertical_lines=[]
    view = View(Item('figure', editor=MPLFigureEditor(),
                           show_label=False),
#                   width=400,
#                   height=300,
                   resizable=True)

    def __init__(self,T):
        super(TracePlot, self).__init__()
        self.T=T
        axes = self.figure.add_subplot(111)
        self.time_steps = np.linspace(0, self.T-1, self.T)
        self.update_plot()
   

    def update_title(self,name):
        axes=self.figure.axes[0]
        axes.set_title(name)
       
    def update_plot(self):
        t = self.time_steps
        if self.y is None:
            y=0*t
        else:
            y=self.y
        axes = self.figure.axes[0]
        
        if not axes.lines:
            axes.plot(t,y)
        else:
            l = axes.lines[0]
            l.set_xdata(t)
            l.set_ydata(y)
            
        axes.relim()
        axes.autoscale_view(True,True,True)
        
        for x in self.vertical_lines:
            axes.axvline(x,color='r',linestyle='--')
        
        canvas = self.figure.canvas
        if canvas is not None:
            canvas.draw()
   
   
   



 