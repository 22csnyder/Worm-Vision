# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 19:09:51 2015

@author: melocaladmin
"""
import numpy as np


class parent:
    def __init__(self):
        print 'parent is init'
    a=5
        
class child(parent):
    b=6
    def __init__(self):
        parent.__init__(self)
        print 'child is init'
        
c=child()












#class abox():
#    pass
#
#
##adjusting flat indicies:
#
##np.logical_and.reduce   ###good stuff
#
#A=np.array([[[0,1,2],[3,4,5],[6,7,8]],[[9,10,11],[12,13,14],[15,16,17]],[[18,19,20],[21,22,23],[24,25,26]]])
#
#idx1=np.array([13,14,16,26])
#
#v=[1,1,0]
#
#idx2=idx1-v[0]*3*3-v[1]*3-v[2]
#
#A2=np.zeros(A.shape)
#A2[:2,:2,:]=A[1:,1:,:]


###############


#def flat2triple(ix):
#    dims=I.shape
#    coord3=ix%dims[2]
#    coord2=( (ix-coord3)%(dims[1]*dims[2]) )/dims[2]
#    coord1=( ix-coord3 - dims[2]*coord2 )/(dims[1]*dims[2])
#    return (coord1,coord2,coord3)








###########
#InspectScene(1)
#sbsContourDraw(phi,Image=I)

#for iii in range(len(PhiRecord)-6,len(PhiRecord)):
#    sbsContourDraw(PhiRecord[iii],Image=I)
#plt.show(block=False)
    

#sbsContourDraw(PhiRecord[0],Image=I)
#sbsContourDraw(PhiRecord[0],Image=I)
#sbsContourDraw(PhiRecord[-1],Image=I)


#sbsContourDraw(phi2[14:20],Image=I[14:20])
#
#squarebysideplot(Itheta,'Itheta')
#squarebysideplot(Ipsi,'Ipsi')


#squarebysideplot(phi2)


##class must be defined above __main__ to work in parallel
#class fit:
#    def __init__(self,N):
#        self.N=N
#    def __call__(self,data):
#        return self.N*data



#if __name__ == '__main__':
#pathWormPics='./../Corral/Snyder/WormPics'
#data_set='150226 Hi-res_40x W5'
#read_dir='./../Corral/Snyder/WormPics'
#working_directory=pathWormPics+'/Synthesized/truncatedStitch/150226 Hi-res_40x/W5'
#stack_path=working_directory+'/stacks\\150226 Hi-res_40x W5.tif'
#W5=Worm(data_set=data_set,read_dir=read_dir,working_directory=working_directory,stack_path=stack_path)
#
#
#stack_path='./../Corral/Snyder/WormPics/150226_CM_Hi-res_40x/W5/stacks\\150226 Hi-res_40x W5.tif'

######
    
    
    
#
#from multiprocessing import Pool as ThreadPool
#from multiprocessing import freeze_support
#n_jobs=8
#
#def do(f):
#    f()
#
#
#def update_by(a):
#    i,stuff=a
#    if i<10:
#        return -1*stuff
#    else:
#        return stuff
#
#
#bag=[]
#class C:
#    data=np.ones(1000)
#    def update(self):
#        stuff=5
#        
#        A=((i,stuff) for i in range(1000))
#
#
#
#
#        pool=ThreadPool(n_jobs)
#        results=pool.map(update_by,A)#results is a list
#        
#        R=np.array(results)
#        
#        C.data+=R
#
#        
#        pool.close()
#        pool.join()
#
#
#if __name__ == '__main__':
#    freeze_support()
#    c=C()
#    c.update()
#    
#    print 'mean is',np.mean(c.data)
#
#    outdata=np.ones(1000)
#    
#    pool=ThreadPool(n_jobs)
#    A=((i,5) for i in range(1000))
#    results=pool.map(update_by,A)



#import matplotlib.pyplot as plt
#import numpy as np
#import math
#
#from WormBox.WormPlot import squarebysideplot
#import pims
#from tifffile import imsave







# Changing the ctf:
#from tvtk.util.ctf import ColorTransferFunction
#ctf = ColorTransferFunction()
## Add points to CTF
#ctf.add_rgb_point(-100, 0, 0, 1)
#ctf.add_rgb_point(0, 0, 0, 0.5)
#ctf.add_rgb_point(50, 0, 0, 0)

#ctf.add_rgb_point(100, 1, 1, 1)
#ctf.add_rgb_point(50, 50, 50, 0.5)
#ctf.add_rgb_point(0, 0.0, 0, 0)
# Update CTF
#volume._ctf = ctf
#volume.update_ctf = True






######WHY can you sometimes use __call__ in parallel and sometimes not!

#
#class WormSaver:
#    def __init__(self,hyper):
#        self.hyper=hyper
#    def __call__(i):
#        worm=hyper.spawn_worm(i)
#        worm.save('Stacks/worm'+str(i))
#
#
#
#class fit:
#    def __init__(self,N):
#        self.N=N
#    def __call__(self,data):
#        return self.N*data
#
#
#
#
#if __name__ == '__main__':
#
#    hyperstack='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Synthesized/150603_1703_Vol_Img_No_Stimulus_W2-2/HyperStack.tif'
#    path='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Synthesized/150603_1703_Vol_Img_No_Stimulus_W2-2'
#    
#    #also works:
#    concatenated='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Synthesized/150603_1703_Vol_Img_No_Stimulus_W2-2/Concatenated Stacks.tif'
#    #wormMovie=Worm4D(concatenated,15)
#    
#    #for i in range(0,5):
#    #    worm=hyper.spawn_worm(i)
#    #    worm.save('Stacks/worm'+str(i))
#    
#    
#    env=Environment(path)
#    
#    hyper=Worm4D('Concatenated Stacks.tif',15,env)
#    
##    save_worm=partial(save_worm_from_hyper,hyper=hyper)
#    
#    
#    n_jobs=2
#    
#    
#    save_worm=WormSaver(hyper)
#    fun=fit(12)
##    joblib_results=Parallel(n_jobs=n_jobs,verbose=1)(delayed(save_worm)(i) for i in range(10))
#    
#    joblib_results=Parallel(n_jobs=n_jobs,verbose=1)(delayed(fun)(i) for i in range(10))
#
#
#





















#
#import numpy
#from mayavi import mlab


#def test_contour3d():
#x, y, z = numpy.ogrid[-5:5:64j, -5:5:64j, -5:5:64j]
#
#scalars = x * x * 0.5 + y * y + z * z * 2.0
#
#obj = mlab.contour3d(scalars, contours=4, transparent=True)
#    return obj


#h=np.ogrid[0:50:1]
#
#mlab.clf()
#x, y, z = np.mgrid[-5:5:64j, -5:5:64j, -5:5:64j]
#values = x*x*0.5 + y*y + z*z*2.0
#mlab.contour3d(values)
#
#
#import skimage.io as io
#P=io.imread('C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Synthesized/truncatedStitch/150226 Hi-res_40x/W5/patch1.tif')
#a,b=np.ogrid[0:5:1,-1:0:10j]
#z,x,y=np.ogrid[0:P.shape[0]:1,0:P.shape[1]:1,0:P.shape[2]:1]
#mlab.contour3d(P)
#
#
#from tvtk.api import tvtk
##from numpy import random
##data = random.random((3, 3, 3))
##i = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0))
##i.point_data.scalars = data.ravel()
##i.point_data.scalars.name = 'scalars'
##i.dimensions = data.shape
#
#def image_data():
#    data = np.random.random((3, 3, 3))
#    i = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0))
#    i.point_data.scalars = data.ravel()
#    i.point_data.scalars.name = 'scalars'
#    i.dimensions = data.shape
#    return i
#    
#mlab.view(image_data())



#investigate pims loading/saving
#def rotate(frame):
#    return np.rot90(frame,3)
#    
#def reorient(frame):
#    return frame[::-1,...]
#def transpose(frame):
#    return np.transpose(frame)
#    
#v=pims.TiffStack(stack,process_func=transpose)
#io.imsave(path+'/pim220_transpose.tiff',v[220])
#
#v=pims.TiffStack(stack,process_func=reorient)
#io.imsave(path+'/pim220_reorrient.tiff',v[220])
#
#v=pims.TiffStack(stack,process_func=rotate)
#io.imsave(path+'/pim220_rotate.tiff',v[220])
#
#v=pims.TiffStack(stack)
#io.imsave(path+'/pim220_no proc func.tiff',v[220])








#class A(object):
#    def __init__(self, i):
#        self.i = i
#
#
#class B(A):
#    def __init__(self, i, j):
#        super(B, self).__init__(i)
#        self.j = j



#def f(x):
#    return x**2
#    
#from multiprocessing.dummy import Pool as ThreadPool
#
#src=range(30)
#
#pool=ThreadPool()
#results=pool.map(f,src)
#pool.close()
#pool.join()













#A=np.hstack([i*np.ones((10,1)) for i in range(10)])



####Using reduce is faster than using np.sum

#def divergence(F):
#    """ compute the divergence of n-D scalar field `F` """
#    return reduce(np.add,np.gradient(F))

######Comparison with kevin-keraudren chan-vese implementation
#Slices=np.load(WormConfig['working directory']+'/'+'Slices.npy')

#205 looks diagonal enough

#img=Slices[205]
#img=Slices[225]
#img=Slices[103]
#img=Slices[68]
#img=Slices[122]
#img=Slices[163]
#img=Slices[40]
#from chanvese import chanvese
#mask = np.zeros(img.shape)
#mask[30:45,30:45] = 1
#chanvese(img,mask,max_its=1000,display=True,alpha=1.0)

###################

###Interesting::Compute contour without plotting anything using undocumented feature:
#from matplotlib import _cntr as cntr#contour
#c=cntr.Cntr(Ixi,Iyi,phi)
#res=c.trace(0.0)
#nseg=len(res)//2
#segments,codes=res[:nseg],res[nseg:]
#
#p=plt.Polygon(segments[0],fill=False,color='red')
#
#fig, ax = plt.subplots(1, 1)
#img_phi = ax.imshow(phi.T, origin='lower')
#plt.colorbar(img_phi)
#ax.hold(True)
#ax.add_artist(p)
#plt.show()


#import collections
#
#Params=collections.OrderedDict([
#'neuron_n':range(1,24)
#,'npoints':16 #must be at least 5 for circle
#,'init_radius':15
#,'alpha':0.0
#,'beta':0.0
#,'h':.1
#,'edge_scale':10.0
#,'n_iter':850
#}
#
#def square(x,*args,**kwargs):
#    if args:
#        print 'args not none'
#        print args
##        for a in args:
##            print a
##    if kwargs:
##        print 'kwargs not none'
##        print kwargs
#
#    if kwargs is not None:
#        for key, value in kwargs.iteritems():
#            print "%s = %s" %(key,value)
#        
##        for k in kwargs:
##            print k
#    return x**2

#li=[(i,np.random.rand(2,3)+i) for i in range(4)]
#scri=[li[3],li[1],li[0],li[2]]

#######make sure you protect your main loops to prevent recursive thread spawn####
#from joblib import Parallel, delayed
#from math import sqrt
#def psq(x):
#    print x,x**2
#Parallel(n_jobs=1)(delayed(psq)(i**2) for i in xrange(10000))
#[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]


#import sys
#if sys.version_info >= (3, 0):
#    sys.stdout.write("Sorry, requires Python 2.x, not Python 3.x\n")
#    sys.exit(1)
#
#import the_real_thing
#if __name__ == "__main__":
#    the_real_thing.main()

#####Here I tried to figure out why matplotlib can't handle showing some3d images
#####    Inconclusive

#%%
#img=Slices[68]
#
#img=img.astype(np.float32)
#img/=img.max()
#
#cimg=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
#
#
#from skimage.color import gray2rgb
#
#
#plt.imshow(img)
#cimg=gray2rgb(img)
#dimg=cimg*2
#plt.imshow(dimg)
#
#
#dst=cimg
#imsave(WormConfig['working directory']+'/'+'msc3.tif',dst,compress=1)
#
#revive_cimg=cv2.imread(WormConfig['working directory']+'/'+'msc3.tif')
#
#sidebysideplot([revive_cimg])
#
#from skimage.viewer import ImageViewer
#view=ImageViewer(cimg)
#view.show()
#
#
#f2='C:/Users/cgs567/Documents/msc/wallpaper/bluelake left 1680x1080.tif'
#
#cwall=cv2.imread(f2)
#
#
#cv2.imshow('wallpap',cwall)
#
#plt.imshow(cwall)
#
#
#patch=np.load(WormConfig['working directory']+'/'+WormConfig['data set identifier']+'.npy')
#
#
#
#
#img=patch[19]
#cimg=gray2rgb(img)
#
#midpoint=np.array([35,35])
#npoints=10
#points=[]
#radius=15
#for i in range(npoints):
#    x=midpoint[0]+radius*math.cos(2*math.pi*i/npoints)
#    y=midpoint[1]+radius*math.sin(2*math.pi*i/npoints)
#    points.append(np.array([int(x),int(y)]))
#points=np.array(points)
#
#drawimg=img.copy()
#
#from itertools import cycle
#def placeLines(drawimg,points):
#    pool=cycle(points)
#    nlines=0
#    p=pool.next()
#    q=pool.next()
#    wt=drawimg.max()*1.5
#    while nlines<npoints:
#        cv2.line(drawimg,(p[0],p[1]),(q[0],q[1]),(wt,0,0))
#        p,q=q,pool.next()
#        nlines+=1
#    return drawimg
#
#
#stack_path='./../Corral/Snyder/WormPics/150226_CM_Hi-res_40x/W5/stacks\\150226 Hi-res_40x W5.tif'
#v=skio.imread(stack_path)#only works in python2 #otherwise need MultiImage class
#
#m=[958, 463, 144]
#neuron_window=np.array([75,75,35])#must be odd
#window_radii=(neuron_window-1)/2#x,y,z
#def get_middle_slice(m=m,v=v):
#    z=m[2]
#    low,high=m-window_radii,m+window_radii+1
#    return v[z,low[0]:high[0],low[1]:high[1]].copy()
#
#sl=v[144,920:980,420:490]
#csl=gray2rgb(sl)
#
#
#
#img=patch[19]
#cimg=gray2rgb(img)
#drawimg=cimg.copy()
#p=(10,50)
#q=(50,50)
#wt=drawimg.max()*1.5
#cv2.line(drawimg,(p[0],p[1]),(q[0],q[1]),(wt,0,0))
#
#draw=placeLines(drawimg,points)












#######Kernel density estimation####

# Author: Jake Vanderplas <jakevdp@cs.washington.edu>
#
#import numpy as np
#import matplotlib.pyplot as plt
#from scipy.stats import norm
#from sklearn.neighbors import KernelDensity
#
#
##----------------------------------------------------------------------
## Plot the progression of histograms to kernels
#np.random.seed(1)
#N = 20
#X = np.concatenate((np.random.normal(0, 1, 0.3 * N),
#                    np.random.normal(5, 1, 0.7 * N)))[:, np.newaxis]
#X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
#bins = np.linspace(-5, 10, 10)
#
#fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
#fig.subplots_adjust(hspace=0.05, wspace=0.05)
#
## histogram 1
#ax[0, 0].hist(X[:, 0], bins=bins, fc='#AAAAFF', normed=True)
#ax[0, 0].text(-3.5, 0.31, "Histogram")
#
## histogram 2
#ax[0, 1].hist(X[:, 0], bins=bins + 0.75, fc='#AAAAFF', normed=True)
#ax[0, 1].text(-3.5, 0.31, "Histogram, bins shifted")
#
## tophat KDE
#kde = KernelDensity(kernel='tophat', bandwidth=0.75).fit(X)
#log_dens = kde.score_samples(X_plot)
#ax[1, 0].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
#ax[1, 0].text(-3.5, 0.31, "Tophat Kernel Density")
#
## Gaussian KDE
#kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(X)
#log_dens = kde.score_samples(X_plot)
#ax[1, 1].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
#ax[1, 1].text(-3.5, 0.31, "Gaussian Kernel Density")
#
#for axi in ax.ravel():
#    axi.plot(X[:, 0], np.zeros(X.shape[0]) - 0.01, '+k')
#    axi.set_xlim(-4, 9)
#    axi.set_ylim(-0.02, 0.34)
#
#for axi in ax[:, 0]:
#    axi.set_ylabel('Normalized Density')
#
#for axi in ax[1, :]:
#    axi.set_xlabel('x')
#
##----------------------------------------------------------------------
## Plot all available kernels
#X_plot = np.linspace(-6, 6, 1000)[:, None]
#X_src = np.zeros((1, 1))
#
#fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
#fig.subplots_adjust(left=0.05, right=0.95, hspace=0.05, wspace=0.05)
#
#
#def format_func(x, loc):
#    if x == 0:
#        return '0'
#    elif x == 1:
#        return 'h'
#    elif x == -1:
#        return '-h'
#    else:
#        return '%ih' % x
#
#for i, kernel in enumerate(['gaussian', 'tophat', 'epanechnikov',
#                            'exponential', 'linear', 'cosine']):
#    axi = ax.ravel()[i]
#    log_dens = KernelDensity(kernel=kernel).fit(X_src).score_samples(X_plot)
#    axi.fill(X_plot[:, 0], np.exp(log_dens), '-k', fc='#AAAAFF')
#    axi.text(-2.6, 0.95, kernel)
#
#    axi.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
#    axi.xaxis.set_major_locator(plt.MultipleLocator(1))
#    axi.yaxis.set_major_locator(plt.NullLocator())
#
#    axi.set_ylim(0, 1.05)
#    axi.set_xlim(-2.9, 2.9)
#
#ax[0, 1].set_title('Available Kernels')
#
##----------------------------------------------------------------------
## Plot a 1D density example
#N = 100
#np.random.seed(1)
#X = np.concatenate((np.random.normal(0, 1, 0.3 * N),
#                    np.random.normal(5, 1, 0.7 * N)))[:, np.newaxis]
#
#X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
#
#true_dens = (0.3 * norm(0, 1).pdf(X_plot[:, 0])
#             + 0.7 * norm(5, 1).pdf(X_plot[:, 0]))
#
#fig, ax = plt.subplots()
#ax.fill(X_plot[:, 0], true_dens, fc='black', alpha=0.2,
#        label='input distribution')
#
#for kernel in ['gaussian', 'tophat', 'epanechnikov']:
#    kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(X)
#    log_dens = kde.score_samples(X_plot)
#    ax.plot(X_plot[:, 0], np.exp(log_dens), '-',
#            label="kernel = '{0}'".format(kernel))
#
#ax.text(6, 0.38, "N={0} points".format(N))
#
#ax.legend(loc='upper left')
#ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), '+k')
#
#ax.set_xlim(-4, 9)
#ax.set_ylim(-0.02, 0.4)
#plt.show()



####Tried to use Chol to do faster inverse. Unsuccessful thus far
#from scipy.linalg import cholesky
#def CholeskyInverse(A):
#    """ 
#    Computes inverse of matrix with Cholesky method
#    """
#    t=cholesky(A)
#    nrows = len(t)
#    B = matzero(nrows, nrows)
# 
#    # Backward step for inverse.
#    for j in reversed(range(nrows)):
#        tjj = t[j][j]
#        S = sum([t[j][k]*B[j][k] for k in range(j+1, nrows)])
#        B[j][j] = 1.0/ tjj**2 - S/ tjj
#        for i in reversed(range(j)):
#            B[j][i] = B[i][j] = -sum([t[i][k]*B[k][j] for k in range(i+1,nrows)])/t[i][i]
#    return B



############Mean shift sklearn example
#import numpy as np
#from sklearn.cluster import MeanShift, estimate_bandwidth
#from sklearn.datasets.samples_generator import make_blobs
#
################################################################################
## Generate sample data
#centers = [[1, 1], [-1, -1], [1, -1]]
#X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)
#
################################################################################
## Compute clustering with MeanShift
#
## The following bandwidth can be automatically detected using
#bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
#
##ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
#ms = MeanShift(bandwidth=.25, bin_seeding=True)
#ms.fit(X)
#labels = ms.labels_
#cluster_centers = ms.cluster_centers_
#
#labels_unique = np.unique(labels)
#n_clusters_ = len(labels_unique)
#
#print("number of estimated clusters : %d" % n_clusters_)
#
################################################################################
## Plot result
#import matplotlib.pyplot as plt
#from itertools import cycle
#
#plt.figure(1)
#plt.clf()
#
#colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
#for k, col in zip(range(n_clusters_), colors):
#    my_members = labels == k
#    cluster_center = cluster_centers[k]
#    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
#    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#             markeredgecolor='k', markersize=14)
#plt.title('Estimated number of clusters: %d' % n_clusters_)
#plt.show()

#########

###Test loading and saving with coordination with fiji###

#def reorient(frame):
#    return frame[::-1,...]
#
#def fiji_pims_reorient(frame):#Lets you pretend that x,y in fiji is row,column in spyder
#    return np.rot90(frame,3)
#
#path='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Synthesized/truncatedStitch/150226 Hi-res_40x/W5/small sample image.tif'
#
##v=pims.TiffStack(filename,process_func=reorient)
#v=pims.TiffStack(path,process_func=fiji_pims_reorient)
#l=[vi for vi in v]
#Y=np.array(v)
#
#imsave("redraw small sample image.tif",Y,compress=1)
#
#
#
#a=np.array([[[1,2,3],[4,5,6],[7,8,9]],[[11,12,13],[14,15,16],[17,18,19]]])
#a=a.astype(np.uint16)
#imsave("3by3by3.tif",a,compress=1)


############
#def close_factors(n):    
#    b=[[i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0]
#    return b[-1]
#    
#def pretty_rect(n,ratio):
#    while True:
#        r,c=close_factors(n)
#        if float(max(r,c))/min(r,c) < ratio:
#            return r,c
#        n+=1
#        
#print pretty_rect(12,1.3)


#class Parent:
#    def __init__(self):
#        self._set_value()
#    def _set_value(self):
#        print 'was called'
#        self.value=5
#class Child(Parent):
#    def __init__(self):
#        Parent._set_value(self)
##        Parent.__init__(self)
##        self._set_value()   
##    def _set_value(self):
##        value=6    
#c=Child()


#import skimage.draw as draw

#Want to make a test dataset

#A=np.zeros((20,40))
#B=A.copy()
#rr3,cc3=draw.circle(5,10,3)
#rr5,cc5=draw.circle(13,18,5)
#A[rr3,cc3]=1
#A[rr5,cc5]=1
#B[rr3,cc3+12]=1
#B[rr5,cc5+12]=1
#
#plt.imshow(A)
#plt.figure()
#plt.imshow(B)


#def square(x,*args,**kwargs):
#    if args:
#        print 'args not none'
#        print args
##        for a in args:
##            print a
##    if kwargs:
##        print 'kwargs not none'
##        print kwargs
#
#    if kwargs is not None:
#        for key, value in kwargs.iteritems():
#            print "%s = %s" %(key,value)
#        
##        for k in kwargs:
##            print k
#    return x**2
#
#d=dict(a=1,b=2)    
#
#print square(2,'d',7,e=3)
#
#def sq(x,**kwargs):
#    FX=kwargs.pop('fx',None)
#    print 'FX is ',FX
##    print kwargs.fromkeys(["fx"])
#    for (key,value) in kwargs.iteritems():
#        print key,value
#    return x**2
#
#
#
#sq(5,fx=6,fy=8)





#
#class Child(Parent):
#    # This does not work
#    def __init__(self):
##        type(self).foobar.extend(["world"])
#        self.foobar.extend(["world"])
#        
#        
#class Parent(object):
#    foobar = ["hello"]
#    c=Child()