# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 09:19:21 2015

@author: cgs567
"""

from WormBox.ioworm import ls,return_those_ending_in
from WormBox.ioworm import getPIMWormData, sort_nicely
from WormBox.WormPlot import squarebysideplot,sidebysideplot,getsquarebysideaxes
import matplotlib.pyplot as plt
import math
import numpy as np
import skimage.io as skio
from WormBox.WormFormat import autoFijiStyleScaling
try:
    from skimage.filters import threshold_otsu,gaussian_filter
except: #is python3
    from skimage.filter import threshold_otsu,gaussian_filter
#import skimage.measure as m #for connected components
from itertools import cycle
from skimage.filters import sobel_h, sobel_v, sobel
from scipy.linalg import circulant
from numpy.linalg import solve
import time
from tifffile import imsave

from WormBox.WormFuns import npinfo #asks for len list, shape nparray within, etc when applicabale
from skimage.transform import rotate

#from joblib import delayed

from WormBox.BaseClasses import Worm,Environment

import scipy.ndimage as ndimage

import time

#WormConfig = {
#    'WormPics directory':pathWormPics,
#    'working directory':pathWormPics+'/Synthesized/truncatedStitch/150226 Hi-res_40x/W5',
#    
#}
#Id=WormConfig['data set identifier']
#Slices=np.load(WormConfig['working directory']+'/'+'Slices.npy')


##########################################################################################
###################################  Define Data  ########################################




#path_tacc='/work/03176/csnyder/Volumes/150226 Hi-res_40x W5/patch1.tif'
#path_windows='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Synthesized/truncatedStitch/150226 Hi-res_40x/W5/patch1.tif'
#path=path_tacc



marker=57


path='/work/03176/csnyder/Volumes/150303_CM_Hi-res_20x/2/patches/patch'+str(marker)+'.tif'
#path='/work/03176/csnyder/Volumes/150226 Hi-res_40x W5/patches/patch'+str(marker)+'.tif'

I=skio.imread(path)
I=I.astype(np.float64)
Iscale=I.max()
I/=Iscale





class SphereParams(object):
    def __init__(self):
#        self.N=8 #number of nonzero frequencies
        self.N=5
        
        self.r0=8.0 #initial radius        
        #tau=0.01
        self.tau=0.001
        self.n_iter=1

        self.r_est=8#needs lamL,Wl[0] nonzero to have effect
#        self.r_est=8#needs lamL,Wl[0] nonzero to have effect

        self.lamP=1.0        
                ###If lamG is too big, will have phi.min() > 0 following level set shrink
                #seems to be working well between 0.01 and 2
        self.lamG=0.0


        self.lamL=10.0#L2 norm penalty        
        self.lamM=40.0
        self.lamR=50.0
        
        #Wl[0]=0#Don't control r0
        
        self.debug=1
        
#        self.delta=3.5#originally 1.2    
#        self.delta=0.7#originally 1.2    
        self.delta=1.2#originally 1.2    
    

def createDictParams(Params):
    DictParams={
    'N':Params.N, #number of nonzero frequencies
    'r0':Params.r0, #initial radius    
    'tau':Params.tau,
    'n_iter':Params.n_iter,
    'r_est':Params.r_est,#needs lamL,Wl[0] nonzero to have effect
    'lamP':Params.lamP,    
            ###If lamG is too big, will have phi.min() > 0 following level set shrink
            #seems to be working well between 0.01 and 2
    'lamG':Params.lamG,    
    'lamL':Params.lamL,#L2 norm penalty    
    'lamM':Params.lamM,#not  yet implemented
    #Wl[0]=0#Don't control r0
    'debug':Params.debug,
    'delta':Params.delta#originally 1.2
    }

eps = np.finfo(np.float).eps

#Need this later
#Wl=np.ones(self.N+1),#include 0 position


#class SphereHarm3d(SphereParams):
#    def __init__(self):
#        SphereParams.__init__(self)
##        super(SphereHarm3d, self).__init__()
#    def fit(self,data):
#        return data*self.N


def mask2phi(mask,sampling):
    dt2=ndimage.distance_transform_bf(mask,'euclidean',sampling)
    dt1=ndimage.distance_transform_bf(1.0-mask,'euclidean',sampling)
    return dt1-dt2+mask.astype('float64') - 0.5

def sbsContourDraw(Contours,idx=None,Image=None):
    colormap='Greens'
    color='red'
    
    if idx is None:
        idx=range(len(Contours))
    n_samples=len(idx)
    fig,axes=getsquarebysideaxes(n_samples)
    
    if Image is None:
        Image=Contours
        colormap='jet'
        color='white'
#    for ax,phi,i in zip(axes,Contours,idx):
    for i,ax in zip(idx,axes):
        titl=str(i)
        phi=Contours[i]
        if hasattr(Image,'__iter__'):
            im=Image[i]
        else:
            im=Image
        ax.imshow(im,cmap=colormap)
        if phi.min()<=0 and phi.max()>=0:
            try:
                ax.contour(phi,0,colors=color)
            except:
                print 'error plotting contour with i=',i
        ax.text(0.1,0.1,titl,verticalalignment='top')

    for ax in axes:
        ax.set_xticklabels([])#has to be done after plot
        ax.set_yticklabels([])
    fig.subplots_adjust(hspace=.001,wspace=0.001)
    fig.tight_layout(pad=0,w_pad=0.0001,h_pad=0.0001)
#    axes[-1].colorbar()
#    plt.colorbar()
    plt.draw()












Params=SphereParams()
N=Params.N
r0=Params.r0
tau=Params.tau
lamP=Params.lamP
n_iter=Params.n_iter
r_est=Params.r_est
lamL=Params.lamL
lamG=Params.lamG
lamM=Params.lamM
lamR=Params.lamR
debug=Params.debug
delta=Params.delta

#        self.Wml=np.ones(self.N+1)#include 0 position   ####Need change for this problem
#Wml=Params.Wml

def getbdry(phi,delta=delta):
    return np.flatnonzero( np.logical_and( phi <= delta, phi >= -delta) )
























#I=np.zeros((70,70,70))
#I[20:40,:,:]=0.5
#I[:,20:40,:]=0.5
#I[:,:,20:40]=0.5
#I[20:40,20:40,20:40]=0.5


#spacing=np.array([3.0,1.0,1.0])#300x100x100um #150226 Hi-res_40x
spacing=np.array([2.0,1.0,1.0])#300x100x100um #150226 Hi-res_40x     z,x,y
#spacing=np.array([1.0,1.0,1.0])





#Init
Izi,Ixi,Iyi=np.indices(I.shape)
#c_est=np.array([10,45,45])
c_est=0.5*np.array(I.shape).astype(np.float)
c=c_est.copy()


Loc=np.vstack([g.ravel() for g in np.indices(I.shape)])

d=np.power(Loc-c_est[:,np.newaxis],2)

distance=np.power(np.sum( np.power(  (Loc-c_est[:,np.newaxis])*spacing[:,np.newaxis]  ,2),axis=0),0.5)
Zero=np.zeros(I.shape)
mask=np.copy(Zero)
mask.flat[distance<=Params.r0]=1.0
phi=mask2phi(mask,sampling=spacing)


Freq=[]
for l in range(1,N+1):#l=0 could be used as radius
    for m in range(-l,l+1):
        Freq.append( (m,l) )
nF=len(Freq)

Aml=np.zeros(nF)
Wml=np.ones(Aml.shape).astype(np.float)
#Bml=np.zeros(nF)

from scipy.special import sph_harm
def Ya(f,angle):#m>0
    m,n=f
    theta,psi=angle
    ans=0.5*( sph_harm(m,n,theta,psi)+sph_harm(-m,n,theta,psi) ) 
    return np.real(ans)

def Yb(f,angle):#m>0
    m,n=f
    theta,psi=angle
    ans=0.5*( sph_harm(m,n,theta,psi)-sph_harm(-m,n,theta,psi) )
    return np.imag(ans)

def Y(freq,angle):
    m,n=freq
    theta,psi=angle
    if m<0:
        return np.sqrt(2)*( -1)**m * np.imag(sph_harm(-m,n,theta,psi))
    elif m==0:
        return np.real(sph_harm(m,n,theta,psi))
    elif m>0:
        return np.sqrt(2)*( -1)**m * np.real(sph_harm(m,n,theta,psi))


##weight matrix:
#Awt=[np.cos(w*Itheta.flat[idx]) for w in Freq]




def cart2polar(x,y,z):
    z*=spacing[0]
    y*=spacing[2]
    x*=spacing[1]
    theta = np.arctan2(y,x)
    xy=np.sqrt(x**2 + y**2)
    d=np.sqrt(x**2 + y**2 + z**2)
    psi=np.arctan2(xy,z)
    return d,theta,psi




#cz,cx,cy=c
#x,y,z=Ixi-cx,Iyi-cy,Izi-cz
#Idist,Itheta,Ipsi=cart2polar(x,y,z)
#Iangle=(Itheta.flat[idx],Ipsi.flat[idx])
#Iangle_=(Itheta,Ipsi)



PhiRecord=[]
PhiRecord.append(phi)

#####STart loop######
for iteration in range(n_iter):
    
    t0=time.time()

    idx=getbdry(phi)
    u1pts = np.flatnonzero(phi<=0)                 # interior points
    u0pts = np.flatnonzero(phi>0)                  # exterior points
    u1 = np.sum(I.flat[u1pts])/(len(u1pts)+eps) # interior mean
    u0 = np.sum(I.flat[u0pts])/(len(u0pts)+eps) # exterior mean
    
    t1=time.time()    
    
    cz,cx,cy=c
    x,y,z=Ixi-cx,Iyi-cy,Izi-cz
    Idist,Itheta,Ipsi=cart2polar(x,y,z)
    Iangle=(Itheta.flat[idx],Ipsi.flat[idx])
    Iangle_=(Itheta,Ipsi)
    
    
#    Awt=[Ya(w, Iangle ) for w in Freq]
    #Bwt=[np.sin(w*Itheta.flat[idx]) for w in Freq]
#    Bwt=[Yb(w, Iangle ) for w in Freq]
    
        
    t2=time.time()
    
    Awt=[Y(w, Iangle ) for w in Freq]
    
    
    cwt=np.hstack([z.flat[idx][:,np.newaxis], x.flat[idx][:,np.newaxis], y.flat[idx][:,np.newaxis]])#Should instead be len 1

    F_P=np.array([ (I.flat[ix]-u1)**2-(I.flat[ix]-u0)**2   for ix in idx])

    distributedF = lamP*F_P






####recentlyadded
#    scaleforce=spacing[0]-np.sin(Ipsi.flat[idx])
#    distributedF*=scaleforce
####recentlyadded

    
    
    delA=np.array([np.sum([wt*force for wt,force in zip(al_wt,distributedF)]) for al_wt in Awt])
    delc=np.sum([wt*f for wt,f in zip(cwt,distributedF)],0) #Should normalize np.linalg.norm(wt)

    delr0=np.sum(distributedF) + lamR * 2.0 * (r0-r_est)###This line is wrong
    
    t3=time.time()    
    
    F_L_a=np.array([2*wl*al for wl,al in zip(Wml,Aml)])   
    F_M=2*(c-c_est) * spacing
    #F_L_b=np.array([2*wl*bl for wl,bl in zip(Wl[1:],Bl)])    
    delA+= lamL * F_L_a        #;delB+= lamL * F_L_b
    delc+=lamM * F_M
    
    t31=time.time()
    
    #F_L_r0=2.0* Wl[0] * (r0-r_est)
    #delr0 += lamL*F_L_r0
    #
    
##new stuff
    N_approx=10
    ind=np.argpartition(np.abs(delA),-N_approx)[:-N_approx]
    delA[ind]=0
##new stuff    
    
    Aml-=tau*delA
#    Bml-=tau*delB*100
    r0-=tau*delr0
    c -=tau*delc


#    if debug==1:
#        print '\niteration ',iteration    
#        print 'r0 is ',r0
#        print 'c is ',c
#        print 'normA is ',np.linalg.norm(Aml)
        
    #Idist,Itheta=cart2polar(Ixi,Iyi,c)
    
    ##update c,r0 at some point
    ##recalculate theta
    cz,cx,cy=c
    x,y,z=Ixi-cx,Iyi-cy,Izi-cz
    Idist,Itheta,Ipsi=cart2polar(x,y,z)
    Iangle=(Itheta.flat[idx],Ipsi.flat[idx])
    Iangle_=(Itheta,Ipsi)


    t32=time.time()

##new stuff
    _R_a=[aml*Y(w,Iangle_) for aml,w in zip(Aml,Freq) if aml!=0]
##new stuff    
#    _R_a=[aml*Y(w,Iangle_) for aml,w in zip(Aml,Freq)]####Most expensive
    
    t33=time.time()
    
    R_a=np.sum(_R_a,0)#Cheap
    
    t34=time.time()
    
#    R_a=np.sum([aml*Y(w,Iangle_) for aml,w in zip(Aml,Freq)],0)####Expensive####

#    t335=time.time()    
    
    R0=r0*np.ones(I.shape)
#    R=R0+R_a+R_b
    R=R0+R_a
    phi=Idist-R
    
#    t4=time.time()    
    tend=time.time()
    print 'Total Iteration Time: ',tend-t0
    print 'Time calculating u0,u1,bdryidx: ', t1-t0
    print 'Time calculating Iangle: ', t2-t1
    print 'Time calculating Dels: ', t3-t2
    print 'Recreating phi for each angle: ',t33-t32
    print 'summing numpy array: ', t34-t33
#    print tend-t34
#    print t31-t3
#    print t32-t31
#    print t33-t32
#    print t335-t33
#    print t4-t335
#    print t32-t0
    
    PhiRecord.append(phi)


from sphereCython.sphhar import cy_Y
from sphereCython.sphhar import py_compute_sphere as pcs

A=np.array([[1,2],[3,4],[5,6]])
A=np.dstack([A,A+6]).astype('double')
Z=np.zeros(A.shape)


A=Ipsi
Z=np.zeros(A.shape)


def old():
    Y((0,3),(0,A))
    
    
def new():
    cy_Y(0,3,Z,A)

#same value for m=0, theta=0
from timeit import Timer

ol_timer=Timer('old()',setup='from __main__ import old')
ne_timer=Timer('new()',setup='from __main__ import new')#17x speedup


ol=Y((0,3),(0,A))
ne=cy_Y(0,3,np.zeros(A.shape),A)
ne=pcs(0,3,0,A.ravel())




#u1pts = np.flatnonzero(phi<=0)                 # interior points
#u0pts = np.flatnonzero(phi>0)                  # exterior points
#u1 = np.sum(I.flat[u1pts])/(len(u1pts)+eps) # interior mean


print '\nMarker ',marker
print 'Interior Intensity: ',u1*Iscale,'\n'




tau*delA
tau*delc
tau*delr0







#phiR0=Idist-R0
#mlab.contour3d(sIzi,sIxi,sIyi,phiR0,color=(0,1,0),opacity=0.25,contours=[0])

####end loop ######


from mayavi import mlab
from mayavi.api import Engine
engine = Engine()
engine.start()
#if len(engine.scenes) == 0:
#    engine.new_scene()
    
#mlab.pipeline.volume(mlab.pipeline.scalar_field(sIzi,sIxi,sIyi,I))
#volume = engine.scenes[0].children[0].children[0].children[0]
#volume.volume_property.scalar_opacity_unit_distance = 3.0
#mlab.contour3d(sIzi,sIxi,sIyi,phi,color=(1,0,0),opacity=0.25,contours=[0])
#volume.volume_mapper.blend_mode = 'maximum_intensity'



sIzi,sIxi,sIyi=spacing[0]*Izi,spacing[1]*Ixi,spacing[2]*Iyi

def InspectScene(n):
    phi=PhiRecord[n]
    engine.new_scene()
    mlab.pipeline.volume(mlab.pipeline.scalar_field(sIzi,sIxi,sIyi,I))
    volume=engine.scenes[-1].children[0].children[0].children[0]
##    #volume.volume_mapper.blend_mode = 'maximum_intensity'
    mlab.contour3d(sIzi,sIxi,sIyi,phi,color=(1,0,0),opacity=0.25,contours=[0])    
    mlab.points3d(spacing[0]*c_est[0],spacing[1]*c_est[1],spacing[2]*c_est[2])

#mlab.title("sddsf")



InspectScene(n_iter)
mlab.text(.05,.02,'150303_CM_Hi-res_20x',width=0.3)
#mlab.text(.05,.02,'150302_CM_Hi-res_40x W5',width=0.3)
mlab.text(.05,.07,'marker '+str(marker),width=0.15)
mlab.text(.05,.10,'iter '+str(n_iter),width=0.08)
dst_dir='/home1/03176/csnyder/temp_output'





#nsaves=0
#150303_CM_Hi-res_20x
#mlab.savefig(dst_dir+'/150302_Hi-res_40x W5 m' +str(marker)+' 0'+str(nsaves)+'.png');nsaves+=1






#    F=distributedF    
#    normF=np.sum(F**2)
#    Basis=[Y(w,Iangle) for aml,w in zip(Aml,Freq)]  
#Basis=[Y(w,Iangle) for w in Freq]  
#conv=[np.sum(b*distributedF) for b in Basis]


    