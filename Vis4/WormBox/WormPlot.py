# -*- coding: utf-8 -*-
"""
Created on Thu Jan 08 11:44:05 2015

@author: melocaladmin
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import math
from mpl_toolkits.mplot3d import axes3d
import pims
import sys
from functools import partial

#from WormFuns import *
from WormBox.WormFuns import *

def showhistogram(X,title=-1):
    plt.figure()
    Xf=X.ravel()
    plt.hist(Xf,bins=100)
    plt.ylim([0,1e5])
    if title!=-1:
        plt.title(title)
    plt.show(block=False)

def showImage(img,title=-1):#BROKEN if you want to call more than once
#    plt.figure()
#    plt.imshow(img,cmap=plt.get_cmap('Greens'),vmax=255)
    fig=plt.imshow(img,cmap=plt.get_cmap('Greens'),vmax=255)
    if title!=-1:
        plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show(block=False)
#    plt.draw()
    return fig
    
def plotDot(idx,ax):
    zvalue=idx[2]
    ax[zvalue].autoscale(False)
    ax[zvalue].plot(idx[1],idx[0],'rD')
    
def getDecentSubplots(img_shape):
    if not img_shape.__len__()==3:print("img_shape in getDecentSubplots must have len=3");return
    ncols=img_shape[0]
    nrows=1
#    y_size,x_size=img_shape[:2]
    y_size,x_size=img_shape[1:]
    
    figscale=y_size*nrows/(x_size*ncols)
    if figscale<1:
        figscale=1
    figsz=7;
    
    if ncols==1:
#        print("only 1 item to be subplotted fyi")
        fig, axes = plt.subplots(nrows,ncols,figsize=(figsz,figsz*figscale),squeeze=False)
        axes=axes[:,0]#magic sauce
    elif ncols>1:
        fig, axes = plt.subplots(nrows,ncols,figsize=(figsz,figsz*figscale))
    for ax in axes:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    
    fig.subplots_adjust(hspace=.001,wspace=0.001)
    fig.tight_layout(pad=0,w_pad=0.0001,h_pad=0.0001)
    return fig, axes
    


       
def sidebysideplot(ListofImages,titles=["no title"],horiz=1,color=['Greens']):
    if not hasattr(titles,'__iter__'):
        titles=[titles]
    ntitles=-1
#    if not isinstance(ListofImages,list):
#        ListofImages=list(ListofImages)
    colormaps=color
    if not isinstance(colormaps,list):colormaps=[colormaps]
    horiz*=-1#I did it wrong sorry not sorry
    if horiz==-1:#User neglected to set to either 1 or 0    
        horiz=1 if ListofImages[0].shape[0]<ListofImages[0].shape[1] else 0    
    
    if set(titles)==set(["no title"]):
        blanktitle=1
    else:
        blanktitle=0
        ntitles=titles.__len__()
    
    try: ncols=ListofImages.__len__()#pims or list
    except:ncols=ListofImages.shape[0]#3dnumpy

#    print "ncols=",ncols
    nrows=1
    y_size,x_size=ListofImages[0].shape[:2]
    figscale=y_size*nrows/(x_size*ncols)
    if figscale<1:
        figscale=1
    figsz=7;
    
    nplots=ncols
    if horiz:
        nrows,ncols=ncols,nrows
#    fig, axes = plt.subplots(nrows,ncols,figsize=(figsz,figsz*figscale))

    if nplots==1:
        fig, axes = plt.subplots(nrows,ncols,figsize=(figsz,figsz*figscale),squeeze=False)
    elif nplots>1:
        fig, axes = plt.subplots(nrows,ncols,figsize=(figsz,figsz*figscale))
#    else:# axes!=-1:
#        for ax in axes:
#            ax.clear()
    for i in range(nplots):
        colormap=colormaps[min(i,len(colormaps)-1)]
#        print i
            
        if nplots==1:
            axes[i,0].imshow(ListofImages[i],cmap=plt.get_cmap(colormap))     
            axes[i,0].set_xticklabels([])
            axes[i,0].set_yticklabels([])
            if blanktitle==0 and i<ntitles:
                axes[i,0].set_title(titles[i])
        elif nplots>=1:
            axes[i].imshow(ListofImages[i],cmap=plt.get_cmap(colormap))     
                
            axes[i].set_xticklabels([])
            axes[i].set_yticklabels([])
            if blanktitle==0 and i<ntitles:
                axes[i].set_title(titles[i])
    fig.subplots_adjust(hspace=.001,wspace=0.001)
    fig.tight_layout(pad=0,w_pad=0.0001,h_pad=0.0001)
#    fig.show()
#    fig.draw()
#    plt.draw()
    fig.canvas.draw()#for when figure already existed
    plt.show(block=False)
    return fig, axes

def close_factors(n):    
    b=[[i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0]
    return b[-1]
    
def pretty_rect(n,ratio):
    while True:
        r,c=close_factors(n)
        if float(max(r,c))/min(r,c) < ratio:
            return r,c
        n+=1

def getsquarebysideaxes(n_images):##This is what you should call to get a grid of axes for nice plotting
    ratio=1.8
    nrows,ncols=pretty_rect(n_images,ratio)
    nplots=nrows*ncols
    
    if nplots==1:
        fig, axes = plt.subplots(nrows,ncols,squeeze=False)
        axes=axes[:,0]
    elif nplots>1:
        fig, axes = plt.subplots(nrows,ncols)
        axes=axes.ravel()
    return fig,axes
        
def squarebysideplot(ListofImages,titles=None,color=['Greens']):
    if not hasattr(titles,'__iter__'):
        if titles is not None:
            titles=[titles]
    colormaps=color
    ratio=1.8
    ntitles=-1
    try:
        n=len(ListofImages)
    except:#not iterable
        n=1
    #    if not isinstance(ListofImages,list):
    #        ListofImages=list(ListofImages)
    if not isinstance(colormaps,list):colormaps=[colormaps]
    
    nrows,ncols=pretty_rect(n,ratio)
    nplots=nrows*ncols
    
    y_size,x_size=ListofImages[0].shape[:2]
    figscale=y_size*nrows/(x_size*ncols)
    if figscale<1:
        figscale=1
    figsz=7;
    
    if nplots==1:
        fig, axes = plt.subplots(nrows,ncols,figsize=(figsz,figsz*figscale),squeeze=False)
        axes=axes[:,0]
    elif nplots>1:
        fig, axes = plt.subplots(nrows,ncols,figsize=(figsz,figsz*figscale))
        axes=axes.ravel()
    
    for i in range(nplots):
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])
    
    
    for i in range(n):       
        colormap=colormaps[min(i,len(colormaps)-1)]                    
        axes[i].imshow(ListofImages[i],cmap=plt.get_cmap(colormap))                        
        try:
                axes[i].set_title(titles[i])
        except:
            pass
    for i in range(n,nplots):
        axes[i].imshow(np.zeros(ListofImages[-1].shape),cmap='Greens')
    fig.subplots_adjust(hspace=.001,wspace=0.001)
    fig.tight_layout(pad=0,w_pad=0.0001,h_pad=0.0001)
    #    fig.show()
    #    fig.draw()
    #    plt.draw()
    fig.canvas.draw()#for when figure already existed
    plt.show(block=False)
    return fig, axes


    
def stackplot(v):
    if isinstance(v,pims.TiffStack):
            z_size=v._count
            y_size, x_size=v[0].shape
    if isinstance(v,np.ndarray):
        z_size,y_size,x_size=v.shape

    
    nrows=4#28 worm per stack
    ncols=7
    
    gs=gridspec.GridSpec(nrows,ncols)
    gs.update(wspace=0.025,hspace=0.025)
    
    figscale=y_size*nrows/(x_size*ncols)
    if figscale<1:
        figscale=1
    figsz=7;
#    fig, axes = plt.subplots(nrows,ncols)
    fig, axes = plt.subplots(nrows,ncols,figsize=(figsz,figsz*figscale))
    #fig, axes = plt.subplots(nrows,ncols,figsize=(3*ncols,3*nrows))
    axes=axes.ravel()
       
    for n in range(z_size):
        axes[n].imshow(v[n],cmap=plt.get_cmap('Greens'),vmin=0,vmax=255)
        axes[n].set_xticklabels([])
        axes[n].set_yticklabels([])
    
    fig.subplots_adjust(hspace=.5,wspace=.001)
    for ax in axes.ravel():
        if not(len(ax.images)):
            fig.delaxes(ax)
    fig.tight_layout(pad=0,w_pad=0,h_pad=0)
    fig.show()

def scatter3d(nzidx):
    fig = plt.figure()
    ax=fig.add_subplot(111,projection='3d',aspect='auto')
    
    x=nzidx[0];y=nzidx[1];z=nzidx[2]
    #print x
    ax.scatter(x,y,z,c='r',marker='o')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')    
    plt.show()

    
def plotClasses(labels,indicies,fig=-1,axes=-1):#feed in "flat" labels and incidies
    if not isinstance(labels,np.ndarray):labels=np.array(labels)
#    if not isinstance(indicies,np.ndarray):indicies=np.array(indicies)
    n_labels=np.unique(labels.flatten()).shape[0]
    
#    idxclass=[(labels==k).nonzero()[0] for k in range(n_labels)]
        
    s_index=list(zip(*indicies))
    z_max=max(s_index[0])
    
#    idxplane=[(s_index[0]==z).nonzero()[0] for z in range(z_max)]    
    
    colormap=plt.cm.gnuplot
    colors=[colormap(ii) for ii in np.linspace(0, 0.9, n_labels)]
    
    img_shape=[max(a)+1 for a in zip(*indicies)]
    if isinstance(fig,int) or isinstance(axes,int):
        fig,axes=getDecentSubplots(img_shape)

    for z in range(z_max):
        for k in range(n_labels):
            axes[z].scatter(s_index[1],s_index[2],c=colors[k],marker='h')
    fig.canvas.draw()

#def plotClasses2(X,labels,indicies,fig=-1,axes=-1):
#    fig,axes=sidebysideplot(patch)
#    n_labels=np.unique(labels).size
#    
#    #s_index=list(zip(*indicies))
#    
#    z_max=max(indicies[0])
#    #z_max=max(s_index[0])
#    #    idxplane=[(s_index[0]==z).nonzero()[0] for z in range(z_max)]    
#    
#    colormap=plt.cm.gnuplot
#    colors=[colormap(ii) for ii in np.linspace(0, 0.9, n_labels)]
#    
#    
#    if isinstance(fig,int) or isinstance(axes,int):
#        img_shape=[max(a)+1 for a in zip(*indicies)]
#        fig,axes=getDecentSubplots(img_shape)
#    
#    for z in range(z_max):
#        for k in range(n_labels):
#            y=indicies[1].take(np.where( (labels==k)*(indicies[0]==z) ))
#            x=indicies[2].take(np.where( (labels==k)*(indicies[0]==z) ))
#            axes[z].scatter(x,y,c=colors[k],marker='h',s=5)
#    fig.canvas.draw()

def nucPatchPlot(patch,indicies):
    
    fig,axes=sidebysideplot(patch)
    z_max=max(indicies[0])
    
    for z in range(z_max+1):
            y=indicies[1].take(np.where( (indicies[0]==z) ))
            x=indicies[2].take(np.where( (indicies[0]==z) ))
            axes[z].scatter(x,y,c='k',marker='h',s=5)
    fig.canvas.draw()

#Handy macro
def imshow(img):
    plt.figure()
    plt.imshow(img)
    plt.colorbar()

#quick black and white plotting
imshow_grey=partial(plt.imshow,cmap=plt.cm.gray_r)

def ezplot(X,title=None):
    plt.figure()
    plt.imshow(X,cmap='Greys')
    plt.colorbar()
    if title:
        plt.title(title)

#################  Table info for later:  #############
#cellDict=the_table.get_celld()
#cellDict[(0,0)].set_width(0.1)

#Normal cell dims seems to be 1.0 x 0.0744   width x height
#font size defaults to 4

#    the_table=ax.table(cellText=cellText,colLabels=colLabels,loc='center',cellLoc='center')

def addTable(fig,ax,**kwargs):
    #**kwargs=cellText=cellText,colLabels=colLabels,loc='center',cellLoc='center')
    ax.clear()#clearing helps table to show up.
    left_col_width=.3
    right_col_width=1-left_col_width
    
    the_table=ax.table(**kwargs)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    the_table.auto_set_font_size(False)    
    the_table.set_fontsize(9.0)
    the_table.scale(1,1.5)
    Keys,Cells=zip(*the_table.get_celld().items())
    for key,cell in the_table.get_celld().items():
        row,col=key
        if col==0:
            cell.set_width(left_col_width)
            cell.set_height(0.1)
        elif col==1:#
            cell._set_text_position('left')
            cell.set_width(right_col_width)
            cell.set_height(0.1)
        else:
            print 'are there more than 2 columns?'
    return the_table


#def plotClass(labels,indicies):
#    n_labels=np.unique(labels.flatten()).shape[0]
#    whereclass=[]
#    for k in range(0,n_labels):
#        onezero=(labels==k)
#        index=onezero.nonzero()
#        x_idx=[];y_idx=[]        
#        for n in range(0,onezero.sum()):
#            i,j=index[0][n],index[1][n]
#            x_idx.append(indicies[1][i])
#            y_idx.append(indicies[2][j])
#        whereclass.append([x_idx,y_idx])    
#    
#    num_plots=labels.shape[0]
#    colormap=plt.cm.gnuplot
#    color=[colormap(ii) for ii in np.linspace(0, 0.9, n_labels)]
#    
##    axis.autoscale(False)
##    for k in range(0,n_labels):
##        axis.scatter(whereclass[k][1],whereclass[k][0],c=color[k],marker='h')
#    for k in range(0,n_labels):
#        plt.scatter(whereclass[k][1],whereclass[k][0],c=color[k],marker='h')
#    plt.show(break=False)
#    