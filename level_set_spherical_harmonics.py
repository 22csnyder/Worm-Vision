# -*- coding: utf-8 -*-
"""
Created on Thu Jun 04 01:03:28 2015

@author: cgs567

This tries to implement
'A Spherical Harmonics Shape Model for Level Set Segmentation'
-Baust and Navab, 2010

Some code borrowed from 
https://github.com/kevin-keraudren/chanvese

"""

from WormBox.ioworm import ls,return_those_ending_in
from WormBox.WormFuns import getRegionBdry,contiguous_image_at_grid
from WormBox.ioworm import getPIMWormData, sort_nicely
from WormBox.WormPlot import *
import matplotlib.pyplot as plt
import math
import numpy as np
import skimage.io as skio
import cv2
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
from joblib import Parallel, delayed
from tifffile import imsave

from WormBox.WormFuns import npinfo #asks for len list, shape nparray within, etc when applicabale
from skimage.transform import rotate
#############################################################################
#####################   Settings and Load Data    ###########################

pathWormPics='./../Corral/Snyder/WormPics'
WormConfig = {
    'WormPics directory':pathWormPics,
    'working directory':pathWormPics+'/Synthesized/truncatedStitch/150226 Hi-res_40x/W5',
    'data set identifier':'150226 Hi-res_40x W5'
}
Id=WormConfig['data set identifier']
Slices=np.load(WormConfig['working directory']+'/'+'Slices.npy')




import scipy.ndimage as ndimage
eps = np.finfo(np.float).eps
def bwdist(a):
    """ 
    this is an intermediary function, 'a' has only True, False vals, 
    so we convert them into 0, 1 values -- in reverse. True is 0, 
    False is 1, distance_transform_edt wants it that way.
    """
    return ndimage.distance_transform_edt(a == 0)
    
def mask2phi(init_a):
    phi = bwdist(init_a)-bwdist(1-init_a)+(init_a) -0.5
    return phi

#from chanvese import *
#max_its=1000
#mask = np.zeros(img.shape)
#mask[30:45,30:45] = 1
#idx = np.flatnonzero( np.logical_and( phi <= 1.2, phi >= -1.2) )

#I=Slices[40]
def showCurveAndPhi(I, phi, color):
    plt.figure()
    myplot = plt.subplot(121)
    myplot.cla()
    axes = myplot.axes
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    
    plt.imshow(I, cmap='gray')
    CS = plt.contour(phi, 0, colors=color) 
    plt.draw()

    myplot = plt.subplot(122)
    axes = myplot.axes
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)       
    plt.imshow(phi)

    plt.draw()

def cart2polar(x, y, c):#must be np.arrays
    d = np.sqrt((x-c[0])**2 + (y-c[1])**2)
    theta = np.arctan2( (y-c[1]), (x-c[0]) )
    return d, theta
    
#Model example:
#(img,(center),(width,height),angle,start_angle,end_angle (deg) , color, thickness))
def sbsContourDraw(Contours,idx=None):
#    if ImageSet==None:
#        ImageSet=Contours
#    titles=map(str,idx)
    if idx is None:
        idx=range(len(Contours))
    n_samples=len(idx)
    fig,axes=getsquarebysideaxes(n_samples)
    
#    for ax,phi,i in zip(axes,Contours,idx):
    for i,ax in zip(idx,axes):
        titl=str(i)
        phi=Contours[i]
        if isinstance(Image,np.ndarray):
            ax.imshow(Image,cmap='Greens')
            ax.contour(phi,0,colors='red')
        else:
            ax.imshow(phi)      
            ax.contour(phi,0,colors='black')
        ax.text(0.1,0.1,titl,verticalalignment='top')

    for ax in axes:
        ax.set_xticklabels([])#has to be done after plot
        ax.set_yticklabels([])
    fig.subplots_adjust(hspace=.001,wspace=0.001)
    fig.tight_layout(pad=0,w_pad=0.0001,h_pad=0.0001)
#    axes[-1].colorbar()
#    plt.colorbar()
    plt.draw()


def takeGrad(f):
    #stack along first axis
    return np.vstack([F[np.newaxis,:] for F in np.gradient(f)])

def normalizeField(F):#return gradients on an image scaled to have norm 1
    norm=np.sqrt( np.sum(F**2,0) )
    norm[norm==0]=1#stopgap measure for dividing by zero #maybe investigate later
    rescale=norm**-1
    return rescale*F

def takeDivergence(F):#assumes dims along first axis
    return np.sum([np.gradient(Fi)[i] for i,Fi in enumerate(F)],0)

def normalized_laplacian(f):
    """ compute the normalized laplacian of n-D scalar field `f` """
    F=takeGrad(f)
    nomoF=normalizeField(F)
    return takeDivergence(nomoF)

def getpsi(phi,c):
    c_idx=vec2idx(c)
    vPhi=takeGrad(phi)
    vPhi[0][c_idx]=1#avoid div by 0
    vPhi[1][c_idx]=1
    v_n=normalizeField(vPhi)    
    rs=np.vstack([ np.cos(Itheta)[np.newaxis,:],np.sin(Itheta)[np.newaxis,:] ])
    v_s=normalizeField(rs)    
    psi=np.sum(v_s*v_n,0) #dotprod
    psi[c_idx]=1.0    
    return psi

def vec2idx(c):
    list_c=[int(cc) for cc in c]
    return tuple(list_c)












###########################################################################
###################           Parameters          #########################

N=30 #number of nonzero frequencies
r0=8.0 #initial radius

#tau=0.01
tau=0.1

#n_iter=1
#n_iter=10
#n_iter=100
#n_iter=146
n_iter=1000

r_est=10#needs lamL,Wl[0] nonzero to have effect
c_est=np.array([37,37])#needs lamM nonzero to have effect


lamP=1.0

        ###If lamG is too big, will have phi.min() > 0 following level set shrink
        #seems to be working well between 0.01 and 2
lamG=0.0

lamL=5.0#L2 norm penalty

lamM=0.0#not  yet implemented

Wl=np.ones(N+1)#include 0 position

#Wl[0]=0#Don't control r0


debug=0


delta=0.7#originally 1.2

###########################################################################
###########################################################################

def getbdry(phi,delta=delta):
    return np.flatnonzero( np.logical_and( phi <= delta, phi >= -delta) )

#I=np.zeros((50,50)).astype('float')


#c=c.astype(np.float)

#cv2.ellipse(I,(int(c[1]),int(c[0])),(21,11),45,0,360,0.5,-1)
#cv2.ellipse(I,(int(c[1]),int(c[0])),(21,11),0,0,180,0.5,-1)
#cv2.ellipse(I,(int(c[1]),int(c[0])),(21,11),0,0,90,0.5,-1)
#cv2.ellipse(I,(int(c[1]),int(c[0])),(11,21),0,0,360,0.5,-1)

#cv2.rectangle(I,(10,30),(40,40),0.5,-1)
#cv2.rectangle(I,(10,20),(40,30),0.5,-1)
#cv2.rectangle(I,(20,10),(30,40),0.5,-1)


###Star shape###
#middle_circle,fat_rect,skinny_rect=I.copy(),I.copy(),I.copy()
#cv2.circle(middle_circle,(int(c[1]),int(c[0])),14,1.0,-1)
#cv2.rectangle(fat_rect,(20,8),(30,42),0.5,-1)
#cv2.rectangle(skinny_rect,(24,1),(26,49),0.5,-1)
#I+= (middle_circle + fat_rect + rotate(fat_rect,90) + rotate(skinny_rect,45) + rotate(skinny_rect,135))
#I[I>0]=0.5
####


I=Slices[225].astype(np.float)
I/=I.max()




c=np.array(I.shape)*0.5
c=c.astype(int)
c=np.array(c).astype(np.float)

c_idx=c



#imshow(rotate(skinny_rect,45))



#I[c_idx]=1.0
#I=gaussian_filter(I,[15,8])
#I/=I.max()

#imshow_grey(I)


###########################################################################
###########################################################################

mask=np.zeros(I.shape)
cv2.circle(mask,(int(c[1]),int(c[0])),int(r0),1.0,-1)
phi=mask2phi(mask)
Ixi,Iyi=np.indices(I.shape)
Idist,Itheta=cart2polar(Ixi,Iyi,c)
Al,Bl=np.array([0]*(N)).astype(np.float),np.array([0]*(N)).astype(np.float)
Freq=np.linspace(1,N,N)
PhiRecord=[]
PhiRecord.append(phi)
    #while not converged

#Get gradient enegery
Igau=gaussian_filter(I,3)
Ig=(1+sobel(Igau)**2)**-1


#imshow_grey(Ig)



















for i in range(n_iter):

    
    ## Alternative method for getting boundary points ##
    ##   Look into more later ##
#    psi=getpsi(phi,c)
#    phi_hat=psi*phi
#    idx=getbdry(phi_hat)
    
    idx=getbdry(phi)
    
    u1pts = np.flatnonzero(phi<=0)                 # interior points
    u0pts = np.flatnonzero(phi>0)                  # exterior points
    u1 = np.sum(I.flat[u1pts])/(len(u1pts)+eps) # interior mean
    u0 = np.sum(I.flat[u0pts])/(len(u0pts)+eps) # exterior mean
    
    #weight matrix:
    Awt=[np.cos(w*Itheta.flat[idx]) for w in Freq]
    Bwt=[np.sin(w*Itheta.flat[idx]) for w in Freq]
    cwt=np.hstack([np.cos(Itheta.flat[idx])[:,np.newaxis] , np.sin(Itheta.flat[idx])[:,np.newaxis]])

    #P
    F_P=np.array([ (I.flat[ix]-u1)**2-(I.flat[ix]-u0)**2   for ix in idx])
    
    #G
    
    NLP=normalized_laplacian(phi)
    I_G=lamG*Ig*NLP#wastefully calculates for entire image first
    F_G=I_G.flat[idx]
   

    distributedF = lamP*F_P + lamG*F_G
    
    
    delA=np.array([np.sum([wt*f for wt,f in zip(al_wt,distributedF)]) for al_wt in Awt])
    delB=np.array([np.sum([wt*f for wt,f in zip(bl_wt,distributedF)]) for bl_wt in Bwt])
    delc=np.sum([wt*f for wt,f in zip(cwt,distributedF)],0)
    delr0=np.sum(distributedF)#This can't be right but the approxi is fine for now.
    
    #L
    F_L_a=np.array([2*wl*al for wl,al in zip(Wl[1:],Al)])    
    F_L_b=np.array([2*wl*bl for wl,bl in zip(Wl[1:],Bl)])    
    delA+= lamL * F_L_a   ;    delB+= lamL * F_L_b
    
    F_L_r0=2.0* Wl[0] * (r0-r_est)
    delr0 += lamL*F_L_r0
       
    Al-=tau*delA
    Bl-=tau*delB
    r0-=tau*delr0
    c -=tau*delc
    
    Idist,Itheta=cart2polar(Ixi,Iyi,c)
    
    #update c,r0 at some point
    #recalculate theta
    
    R_a=np.sum([al*np.cos(w*Itheta) for al,w in zip(Al,Freq)],0)
    R_b=np.sum([bl*np.sin(w*Itheta) for bl,w in zip(Bl,Freq)],0)
    R0=r0*np.ones(I.shape)    
    R=R0+R_a+R_b
    phi=Idist-R
    
    
    #Sometimes phi.min() becomes greater than 0
    #check if this happens, then go back to previous state and decrease tau    
    
    
    PhiRecord.append(phi)

#    showCurveAndPhi(I,phi,'yellow')











#c -= 10*delc
#_,theta=cart2polar(Ixi,Iyi,c)
#plt.figure()
#plt.imshow(theta)











#################################

n_samples=19
inspect_idx=np.logspace(0,np.log10(n_iter),n_samples).astype('int')
if n_iter<n_samples:
    inspect_idx=np.linspace(0,n_iter,n_samples).astype('int')
inspect_idx=np.unique(inspect_idx)

#Sometimes doesn't work??
sbsContourDraw(PhiRecord,inspect_idx)

sbsContourDraw(PhiRecord,inspect_idx,Image=I)



#squarebysideplot([PhiRecord[ii] for ii in inspect_idx],'jet')

#Inspect what in the world is happenning at 10 iterations

#phi=PhiRecord[-1]
#dst=np.zeros(I.shape)
##idx = np.flatnonzero( np.logical_and( phi <= 1.2, phi >= -1.2) )
#dst.flat[idx]=0.5
#plt.imshow(dst)
#plt.contour(phi,0,colors='black')

if debug==1:
    ####### Inspect Forces #######
    J=I.copy()
    J.flat[idx]+=0.3
    distForce=np.zeros(I.shape)
    distForce.flat[idx]=distributedF    
    sidebysideplot([J,distForce],['Overlay','forces'])

    
    ####### Inspect divergence #######
    local_F_G=np.zeros(I.shape)
    nlp=normalized_laplacian(phi)
    Force_G=Ig*nlp
    local_F_G.flat[idx]=Force_G.flat[idx]
    local_F_G.flat[idx]*=Ig.flat[idx]
    #fig,axes=sidebysideplot([I,phi,nlp,Ig,Ig*nlp,local_F_G],['I','phi','nlp','Ig','Ig*nlp','local_F_G'])
    fig,axes=squarebysideplot([I,phi,nlp,Ig,Ig*nlp,local_F_G],['I','phi','nlp','Ig','Ig*nlp','local_F_G'],'Greys')
    for ax in axes:
        ax.contour(phi,0,colors='black')
    plt.draw()

    ####### Inspect boundary ####### 
    phi_hat=psi*phi
    bdry,bdry2=np.zeros(I.shape),np.zeros(I.shape)
    idx =getbdry(phi)
    idx2=getbdry(phi_hat)
    bdry.flat[idx]=0.5
    bdry2.flat[idx2]=0.5
    #squarebysideplot([bdry,bdry2],[' ','hat'],'Greys')
    sidebysideplot([bdry,bdry2],[' ','hat'],color='Greys')
    bd2=np.zeros(I.shape)


#fig,axes=plt.subplots(1,2)
#im=axes[0].imshow(Awt[0].reshape(50,50))
#cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
#fig.colorbar(im, cax=cbar_ax)
#axes[1].imshow(Bwt[0].reshape(50,50))
#plt.show()

########################################
#o=np.array([[0,0,0,0],[1,2,2,1],[1,2,3,2],[1,1,2,1]])
#O=takeGrad(o)
#Div=np.sum([np.gradient(Fi)[i] for i,Fi in enumerate(O)],0)

##########Investigate offset phi0####
    
    
#Coeff=np.vstack([ Al[np.newaxis,:] ,Bl[np.newaxis,:] ])
#phi0=math.pi/4
#Rot_phi0=np.array([[np.cos(phi0),np.sin(phi0)],[-np.sin(phi0),np.cos(phi0)]])
##Rotate:
#NewCoeff=np.dot(Rot_phi0,Coeff)
#rotAl,rotBl=NewCoeff
#R_a=np.sum([al*np.cos(w*Itheta) for al,w in zip(rotAl,Freq)],0)
#R_b=np.sum([bl*np.sin(w*Itheta) for bl,w in zip(rotBl,Freq)],0)
#R0=r0*np.ones(I.shape)    
#R=R0+R_a+R_b    
#phi=Idist-R
#showCurveAndPhi(I,phi,'red')
#plt.title('plus phi0=45deg')

#########

#plt.figure()
#plt.imshow(Itheta)
#plt.title('theta')
#plt.colorbar()
#
#fig,axes=plt.subplots
#plt.imshow(np.cos(Itheta))
#plt.title('cos')
#plt.colorbar()




#myplot = plt.subplot(121)
#myplot.cla()
#axes = myplot.axes
#axes.get_xaxis().set_visible(False)
#axes.get_yaxis().set_visible(False)
#
#plt.imshow(I, cmap='gray')
#CS = plt.contour(phi, 0, colors=color) 
#plt.draw()
#
#myplot = plt.subplot(122)
#axes = myplot.axes
#axes.get_xaxis().set_visible(False)
#axes.get_yaxis().set_visible(False)       
#plt.imshow(phi)
#
#plt.draw()


#    cos_wt_1=[-1*math.cos(th) for th in theta]
#    sin_wt_1=[-1*math.sin(th) for th in theta]


####################fun for messing around #############


##R=r0*np.ones(I.shape)+10*np.cos(Itheta)+2*np.sin(Itheta)
#R0,R1,R2,R3=[np.zeros(I.shape)]*4
#
##R0=r0*np.ones(I.shape)
##R1=2*np.cos(Itheta)+2*np.sin(Itheta)
##R2=3*np.cos(2*Itheta)+3*np.sin(2*Itheta)
#R3=2*np.cos(3*Itheta)+4*np.sin(3*Itheta)
#
#R=R0+R1+R2+R3
#test_phi=di-R
#
#showCurveAndPhi(I,test_phi,'red')



############################# Debug and sample code below ##################################


#imshow_grey(mask)
#a1,b1,a2,b2=0,0,0,0

#Better make sure that this strip doesn't skip parts of the sphere somehow

#for it in range(20):
#    idx = np.flatnonzero( np.logical_and( phi <= 1.2, phi >= -1.2) )
#    
#    
#    u1pts = np.flatnonzero(phi<=0)                 # interior points
#    u0pts = np.flatnonzero(phi>0)                  # exterior points
#    u1 = np.sum(I.flat[u1pts])/(len(u1pts)+eps) # interior mean
#    u0 = np.sum(I.flat[u0pts])/(len(u0pts)+eps) # exterior mean
#    
#    #F = (I.flat[idx]-u1)**2-(I.flat[idx]-u0)**2    # force from image information
#    #Break force up into components
#    
#    xi,yi=np.indices(I.shape)
#    
#    theta=[math.atan2(y-c[1],x-c[0]) for y,x in zip(yi.flat,xi.flat)]
#    
#    cos_wt_1=[-1*math.cos(th) for th in theta]
#    sin_wt_1=[-1*math.sin(th) for th in theta]
#    
#    cos_wt_2=[-1*math.cos(2*th) for th in theta]
#    sin_wt_2=[-1*math.sin(2*th) for th in theta]
#    distributedF=[ (I.flat[ix]-u1)**2-(I.flat[ix]-u0)**2   for ix in idx]
#    
#    Fa1=sum([w*f for (w,f) in zip(cos_wt_1,distributedF)])
#    Fb1=sum([w*f for (w,f) in zip(sin_wt_1,distributedF)])
#    
#    Fa2=sum([w*f for (w,f) in zip(cos_wt_2,distributedF)])
#    Fb2=sum([w*f for (w,f) in zip(sin_wt_2,distributedF)])
#    
#    h=0.1
#    a1+=h*Fa1
#    b1+=h*Fb1
#    a2+=h*Fa2
#    b2+=h*Fb2
#    
#    print 'vals are'
#    print a1,b1,a2,b2
#    
#    
#        
#    Ixi,Iyi=np.indices(I.shape)
#    
#    di,Itheta=cart2polar(Ixi,Iyi,c)
#    r0=15
#    R=r0*np.ones(I.shape) + a1*np.cos(Itheta)+a2*np.cos(2*Itheta) + b1*np.sin(Itheta)+b2*np.sin(2*Itheta)
#    
#    phi=di - R
#    
#    
#    showCurveAndPhi(I,phi,'yellow')



#dst=np.zeros(I.shape)
#dst.flat[idx]=distributedF
#squarebysideplot([I,mask,phi],['I','init mask','phi'])
#imshow_grey(dst)
#plt.colorbar()
#plt.title('Force on phi')




#####################################################################################


#for loc in idx


####Previous implementation to sample from:

#display=1
#
#if display:
#    if np.mod(its,50) == 0:            
#        print 'iteration:', its
#        showCurveAndPhi(I, phi, color)
#
#else:
#    if np.mod(its,10) == 0:
#        print 'iteration:', its
#
##-- find interior and exterior mean
#upts = np.flatnonzero(phi<=0)                 # interior points
#vpts = np.flatnonzero(phi>0)                  # exterior points
#u = np.sum(I.flat[upts])/(len(upts)+eps) # interior mean
#v = np.sum(I.flat[vpts])/(len(vpts)+eps) # exterior mean
#
#F = (I.flat[idx]-u)**2-(I.flat[idx]-v)**2    # force from image information
#curvature = get_curvature(phi,idx)  # force from curvature penalty
#
#dphidt = F /np.max(np.abs(F)) + alpha*curvature # gradient descent to minimize energy
#
##-- maintain the CFL condition
#dt = 0.45/(np.max(np.abs(dphidt))+eps)
#
##-- evolve the curve
#phi.flat[idx] += dt*dphidt
#
##-- Keep SDF smooth
#phi = sussman(phi, 0.5)
#
#new_mask = phi<=0
#c = convergence(prev_mask,new_mask,thresh,c)
#
#if c <= 5:
#    its = its + 1
#    prev_mask = new_mask
#else: stop = True





