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

from itertools import product,chain

from WormBox.BaseClasses import Worm,Environment

import scipy.ndimage as ndimage

import time

tbeginning=time.time()


#path_tacc='/work/03176/csnyder/Volumes/150226 Hi-res_40x W5/patch1.tif'
#path=path_tacc


#path_windows='C:/Users/cgs567/Documents/Corral/Snyder/WormPics/Ki-Confocal/150226_CM_Hi-res_40x/W5/stacks/patches/patch115.tif'


marker=116
#marker=57



path='/work/03176/csnyder/Volumes/150303_CM_Hi-res_20x/2/patches/patch'+str(marker)+'.tif'
#path='/work/03176/csnyder/Volumes/150226 Hi-res_40x W5/patches/patch'+str(marker)+'.tif'

I=skio.imread(path)
#I=np.zeros((70,70,70))
#I[30:40,30:40,30:40]=0.5

#I[20:40,:,:]=0.5
#I[:,20:40,:]=0.5
#I[:,:,20:40]=0.5
#I[20:40,20:40,20:40]=0.5
#I=np.zeros((3,3,3))
#I=np.array([[[0,1,2],[3,4,5],[6,7,8]],[[9,10,11],[12,13,14],[15,16,17]],[[18,19,20],[21,22,23],[24,25,26]]])



I=I.astype(np.float64)
Iscale=I.max()
I/=Iscale


#spacing=np.array([3.0,1.0,1.0])#300x100x100um #150226 Hi-res_40x
#spacing=np.array([2.0,1.0,1.0])#300x100x100um #150226 Hi-res_40x
spacing=np.array([1.0,1.0,1.0])

c_est=0.5*np.array(I.shape).astype(np.float)#z,x,y
Center=c_est.copy()

debug=0
debug_membership=0
debug_time=1

theta_steps=40
phi_steps=20
class SphereParams(object):
    def __init__(self):
#        self.N=8 #number of nonzero frequencies
        self.N=5
        
        self.r0=8.0 #initial radius
#        self.r0=6.0
        
        #tau=0.01
        self.tau=0.001
        self.n_iter=4

        self.r_est=8.0#needs lamL,Wl[0] nonzero to have effect
#        self.r_est=8#needs lamL,Wl[0] nonzero to have effect

        self.lamP=1.0 
                ###If lamG is too big, will have phi.min() > 0 following level set shrink
                #seems to be working well between 0.01 and 2
        self.lamG=0.0


        self.lamL=1.0#L2 norm penalty        
        self.lamM=1.0
        self.lamR=1.0
        
        #Wl[0]=0#Don't control r0
        
#        self.debug=1
        
#        self.delta=3.5#originally 1.2    
#        self.delta=0.7#originally 1.2    
        self.delta=1.2#originally 1.2



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
#debug=Params.debug
delta=Params.delta

#        self.Wml=np.ones(self.N+1)#include 0 position   ####Need change for this problem
#Wml=Params.Wml

def getbdry(phi,delta=delta):
    return np.flatnonzero( np.logical_and( phi <= delta, phi >= -delta) )



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
        
        

def cart2polar(x_,y_,z_,spacing):
    z=z_*spacing[0]
    y=y_*spacing[2]
    x=x_*spacing[1]
    theta = np.arctan2(y,x)
    xy=np.sqrt(x**2 + y**2)
    d=np.sqrt(x**2 + y**2 + z**2)
    psi=np.arctan2(xy,z)
    return d,theta,psi

def cart2_Discrete_polar(x_,y_,z_,spacing,theta_steps,phi_steps):
    z=z_*spacing[0]
    x=x_*spacing[1]
    y=y_*spacing[2]
    theta =np.round( np.arctan2(y,x)*theta_steps/2/np.pi +theta_steps/2 )%theta_steps #Returns theta in (0,1,2,..,theta_steps-1)
    xy=np.sqrt(x**2 + y**2)
    d=np.sqrt(x**2 + y**2 + z**2)
    psi=np.round( np.arctan2(xy,z) * (phi_steps-1) /np.pi  )#Returns phi in (0,1,2,...,phi_steps)
    return d,theta.astype('int'),psi.astype('int')


#Reference is a lookup table for transforming cartesian coordinates to the radius for the nearest corresp angle
#You can use Reference Grid to use extra resolution or some other transformation



class ReferenceGrid():
    def __init__(self,I,spacing):
        self.spacing=spacing
        self.I=I
#        self.ref_image=np.zeros(2*np.array(I.shape)+1)
        self.ref_image=np.zeros(np.array(I.shape))#even for now... bleh        
        self.origin=np.array(self.ref_image.shape)//2
        self.z,self.x,self.y=np.indices(self.ref_image.shape)-self.origin[:,np.newaxis,np.newaxis,np.newaxis]
        self.d,self.th,self.ph=cart2polar(self.x,self.y,self.z,spacing)
        self.Distance=np.sqrt( (self.z*spacing[0])**2 + ((self.x)*spacing[1])**2  +  ((self.y)*spacing[2])**2  )
        
        ##new##
        self.ref_size=self.ref_image.size
        self.ref_shape=np.array(self.ref_image.shape).astype(np.int)
        
    def set_discrete_angles(self,theta_steps,phi_steps):
        self.d,self.di_th,self.di_ph=cart2_Discrete_polar(self.x,self.y,self.z,self.spacing,theta_steps,phi_steps)
        self.AngleIdx_Flat=[]
        self.AngleIdx=[]
        for t,p in product(range(theta_steps),range(phi_steps)):
            self.AngleIdx_Flat.append( np.flatnonzero( np.logical_and( self.di_th == t, ref.di_ph == p) ) )
            self.AngleIdx.append(         np.nonzero( np.logical_and( self.di_th == t, ref.di_ph == p) ) )
        self.dist_at_angle=[self.Distance.flat[anxf] for anxf in self.AngleIdx_Flat]

        self.Dcrt_Angle=-1*np.ones(self.ref_image.shape)
        for a in xrange(len(self.AngleIdx_Flat)):
            self.Dcrt_Angle.flat[self.AngleIdx_Flat[a]]=a
        self.Dcrt_Angle=self.Dcrt_Angle.astype(np.int)

    def getCorrespondence(self,Angle):#outputs a map from cart grid to closest point on spherical grid
        theta_grid=Angle[0][:,0]
        phi_grid=Angle[1][0,:]
        theta_steps=len(theta_grid)
        phi_steps=len(phi_grid)
        
        MapTheta,MapPhi=np.zeros(self.ref_image.shape),np.zeros(self.ref_image.shape)
#could enumerate here to get the flat indicies
#        for i,x,y,z in enumerate(zip(self.z.flat+self.origin[0],self.x.flat+self.origin[1],self.y.flat+self.origin[2])):
        for i in xrange(self.ref_image.size):
            MapTheta.flat[i]=np.round(self.th.flat[i]*theta_steps/2/np.pi + theta_steps/2.0)%theta_steps
            MapPhi.flat[i]=np.round(self.ph.flat[i]*(phi_steps-1)/np.pi)
            
#            MapTheta.flat[i]=np.where( theta_grid*theta_steps/2/np.pi==np.round(self.th[x,y,z]*theta_steps/2/np.pi) )[0][0]#20
#            MapPhi[x,y,z]=np.where( phi_grid*(phi_steps-1)/np.pi==np.round(self.ph[x,y,z]*(phi_steps-1)/np.pi) )[0][0]
        return (MapTheta,MapPhi)
    
    def toReferenceCoordinates(self,x,y,z,Center):
        pos=np.array([z,x,y])
        ref_pos=np.round( pos-Center+self.origin )    ####Put scale here if want super resolution
        out_z,out_x,out_y=ref_pos
        return out_x,out_y,out_z

Freq=[]
for l in range(1,N+1):#l=0 could be used as radius
    for m in range(-l,l+1):
        Freq.append( (m,l) )
nF=len(Freq)

#Theta 0,2pi    phi 0,pi


Izi,Ixi,Iyi=np.indices(I.shape)

####Discretize####
#theta_steps=#def above
#phi_steps=#def above
theta_spacing = 2*np.pi/theta_steps
phi_spacing=np.pi/(phi_steps-1)
theta_grid=np.mgrid[-np.pi:np.pi:theta_spacing]
phi_grid=np.mgrid[0:np.pi:np.complex(0,phi_steps)]

def AngleIterator():
    return product(range(theta_steps),range(phi_steps))

ref=ReferenceGrid(I,spacing)  #Do this later. Hope radius doesn't hit boundary
ref.set_discrete_angles(theta_steps,phi_steps)


ref_dims=np.array(ref.ref_image.shape)
std_dims=np.array(I.shape)

#def shift_idx_ref2std_flat(idx,Center):
def ref3d_2_stdflat(idx,Center):
    if not idx[0].size:
        return np.empty(0)
    Center=np.round(Center)###May change later to allow more resolution
    v_displace =  np.round(Center) - ref.origin #z,x,y
    new_idx=[ ix+vd for ix,vd in zip(idx,v_displace) ]#for each dim
    valid_ones=np.logical_and.reduce((new_idx[0]>=0,new_idx[0]<std_dims[0],new_idx[1]>=0,new_idx[1]<std_dims[1],new_idx[2]>=0,new_idx[2]<std_dims[2]))
    if True in valid_ones:
        final_idx=[nx[valid_ones] for nx in new_idx]#for each dim
        return np.array( [n0*std_dims[1]*std_dims[2] + n1*std_dims[2] + n2   for n0,n1,n2 in zip(final_idx[0],final_idx[1],final_idx[2]) ]).astype('int')#convert to flat
    else:
        print "Warning: Center has moved far to the side. no corresponding indicies in std system from ref system"
        return np.empty(0)
    


#theta_grid,phi_grid=np.mgrid[0:2*np.pi:theta_spacing,0:np.pi:np.complex(0,phi_steps)]

Angle=np.mgrid[-np.pi:np.pi:theta_spacing,0:np.pi:np.complex(0,phi_steps)]
Radius=np.zeros(Angle[0].shape)+r0 #theta_steps x phi_steps 

BasisFunctions=[Y(ml,Angle) for ml in Freq]

Aml=np.zeros(nF)
Wml=np.ones(Aml.shape).astype(np.float)

mapTheta,mapPhi=ref.getCorrespondence(Angle)#z,x,y #tricky

CenterRecord=[]
tmp=np.copy(Center)
CenterRecord.append(tmp)
RadiusRecord=[]
RadiusRecord.append(Radius.copy())
#PhiRecord=[]
#PhiRecord.append(phi)


Itheta=[];Iphi=[]    #ugly patch to get angles
for dcrt_theta,dcrt_phi in AngleIterator():     
    Itheta.append( theta_grid[dcrt_theta] )
    Iphi.append( phi_grid[dcrt_phi])
Itheta=np.array(Itheta)
Iphi=np.array(Iphi)
Iangle=(Itheta,Iphi)
Awt=np.array([Y(w, Iangle ) for w in Freq]) #Moved out of loop


###New stuff###
std_size=I.size
std_shape=np.array(I.shape).astype(np.int_)

t01=0
t02=0
t03=0
t04=0
t05=0
t06=0
t07=0
t08=0
t09=0
t010=0
t011=0
t012=0
t013=0
t014=0









from polar_nucleus3D_Cython.cy_sphere_iter import cy_iteration

'''
def dum():
    a,b,c,d=cy_iteration(Radius,I,std_size,ref.ref_size,ref.ref_shape.astype(np.int),
    std_shape.astype(np.int),ref.Dcrt_Angle,ref.Distance,Awt,np.array(Awt.shape),
    Aml,Wml,Center,c_est,r0,r_est,ref.origin,
    lamP,lamM,lamR,lamL,delta,tau,n_threads)     
import timeit
cyiter=timeit.Timer('dum()',setup='from __main__ import dum')
#cyiter.timeit(500)/500.0
'''

tminus1=time.time()
print 'Time till start of loop: ',tminus1-tbeginning

#####STart loop######


#if n_iter==1:
#    debug_membership=1

for iteration in range(n_iter):
    ####cython stufff####
    n_threads=8
    cy_u1,cy_delA,cy_delc,cy_delr0=cy_iteration(Radius,I,std_size,ref.ref_size,ref.ref_shape.astype(np.int),
    std_shape.astype(np.int),ref.Dcrt_Angle,ref.Distance,Awt,np.array(Awt.shape),
    Aml,Wml,Center,c_est,r0,r_est,ref.origin,
    lamP,lamM,lamR,lamL,delta,tau,n_threads)    
    print 'cython output:'
    print 'cy_u1 ',cy_u1
    print 'cy_delc ',cy_delc
    print 'cy_delr0 ',cy_delr0    
    ##############
    
    
    t0=time.time()    
    centered_Izi=Izi-Center[0];  centered_Ixi=Ixi-Center[1];   centered_Iyi=Iyi-Center[2]    
#    Distance=np.sqrt( (centered_Izi*spacing[0])**2 + ((centered_Ixi)*spacing[1])**2  +  ((centered_Iyi)*spacing[2])**2  )    
    t_areyoukiddingme=time.time()

####For each discrete angle, grab stuff of the appropriate bleh####
    BoundaryPointsPerAngle=[]
    u0=0; cnt_outside=0
    u1=0; cnt_inside=0    
    idx_at_angle=[]
    sum_bdry_intensity_at_angle=[]
    n_points_on_bdry_at_angle=[]#maybe do something with this later to adjust for nonuniform boundary segment
    
#    Itheta=[];Iphi=[]    #ugly patch
    
    for i,(dcrt_theta,dcrt_phi) in enumerate(AngleIterator()):##Iterate over every slice of angle
        
        t01=time.time()
#        Itheta.append( theta_grid[dcrt_theta] )
#        Iphi.append( phi_grid[dcrt_phi])
        t02+=time.time()-t01
        R=Radius[dcrt_theta,dcrt_phi]
        t03+=time.time()-t01
#        outside_idx =[ ix[ R < ref.Distance.flat[ref.AngleIdx_Flat[i]] ] for ix in ref.AngleIdx[i] ]
        t04+=time.time()-t01
        
        
        #Let's just skip this section for speed: Call it u0=0.05

#        std_flat_outside_idx= ref3d_2_stdflat( outside_idx, Center)
#        if std_flat_outside_idx.size:#checks if not empty
#            u0+=np.sum(I.flat[std_flat_outside_idx])
#            cnt_outside+=len(std_flat_outside_idx)
#        else:
#            if debug_membership:
#                print 'angle index ',i,' has empty outside at d_th=',dcrt_theta,' d_ph=',dcrt_phi

        t05+=time.time()-t01
#        inside_idx  =[ ix[ R > ref.Distance.flat[ref.AngleIdx_Flat[i]] ] for ix in ref.AngleIdx[i] ]    #Costly
        inside_idx  =[ ix[ R > ref.dist_at_angle[i] ] for ix in ref.AngleIdx[i] ]    #Costly
        t06+=time.time()-t01
        std_flat_inside_idx = ref3d_2_stdflat(  inside_idx, Center)    #Costly
        t07+=time.time()-t01
        if std_flat_inside_idx.size:
            u1+=np.sum(I.flat[std_flat_inside_idx])
            cnt_inside+=len(std_flat_inside_idx)
        else:
            if debug_membership:
                print 'angle index ',i,' has empty inside at   d_th=',dcrt_theta,' d_ph=',dcrt_phi
        t08+=time.time()-t01
        #Costly next line
#        bdry_idx=[ ix[np.logical_and( R-ref.Distance.flat[ref.AngleIdx_Flat[i]] <= delta , ref.Distance.flat[ref.AngleIdx_Flat[i]] - R <= delta)]   for ix in ref.AngleIdx[i] ]
        bdry_idx=[ ix[np.logical_and( R-ref.dist_at_angle[i] <= delta , ref.dist_at_angle[i] - R <= delta)]   for ix in ref.AngleIdx[i] ]
        t09+=time.time()-t01
        std_flat_bdry_idx=ref3d_2_stdflat( bdry_idx, Center)
        t010+=time.time()-t01
        idx_at_angle.append(std_flat_bdry_idx)
        t011+=time.time()-t01
        if std_flat_bdry_idx.size:
            sum_bdry_intensity_at_angle.append(np.sum(I.flat[std_flat_bdry_idx]))
            n_points_on_bdry_at_angle.append(len(std_flat_bdry_idx))
        else:
            n_points_on_bdry_at_angle.append(0)
            if debug_membership:
                print 'angle index ',i,' has empty boundary at d_th=',dcrt_theta,' d_ph=',dcrt_phi
        
#        if std_flat_bdry_idx.size:    
#            break
        t012+=time.time()-t01
    
#    print "u1 before divide",u1
    u1/=cnt_inside
#    u0/=cnt_outside ######skipped for speed!
    u0=0.05
        
#        a=stoooooop_at_first_iter______________________________
#    t013=time.time()#-t012      
        
#    Itheta=np.array(Itheta)
#    Iphi=np.array(Iphi)
#    Iangle=(Itheta,Iphi)    
    
#    t014=time.time()#-t014  
    
    idx=list(chain(*idx_at_angle))#conforms to old notation
    
    dcrt_I=np.zeros(len(idx_at_angle))
    for ang,ix in enumerate(idx_at_angle):
        if ix.size:
            dcrt_I[ang]=np.mean(I.flat[ix])
            
            
    t1=time.time()


#    a=stoooooop

    
    #Nothing needed to do here
    
    t2=time.time()
    
#    Awt=[Y(w, Iangle ) for w in Freq]
#    Awt=[Y(w, Iangle ) for w in Freq]#can be moved out of loop
    t21=time.time()
    cwt=np.hstack([centered_Izi.flat[idx][:,np.newaxis], centered_Ixi.flat[idx][:,np.newaxis], centered_Iyi.flat[idx][:,np.newaxis] ])
    t22=time.time()
    F_P=np.array([ (I.flat[ix]-u1)**2-(I.flat[ix]-u0)**2   for ix in idx])#wrong
    distributedF = lamP*F_P
    
    t23=time.time()

#    dcrt_F_P=np.array([ (I.flat[ix]-u1)**2-(I.flat[ix]-u0)**2   for ix in idx])

#    dcrt_F_P= (dcrt_I-u1)**2  - (dcrt_I-u0)**2 #Maybe not fully equivalent but close
#    dcrt_F_P=  [np.sum((I.flat[ix]-u1)**2-(I.flat[ix]-u0)**2)   for ix in idx_at_angle]#Each ix here is an array of pix at that angle
    dcrt_F_P=np.zeros(len(idx_at_angle))#loop is eq to above
    for a,ix in enumerate(idx_at_angle):
        if ix.size:
            dcrt_F_P[a]=np.sum((I.flat[ix]-u1)**2-(I.flat[ix]-u0)**2)

    dcrt_distributedF= lamP*dcrt_F_P

    t24=time.time()
####recentlyadded
#    scaleforce=spacing[0]-np.sin(Ipsi.flat[idx])
#    distributedF*=scaleforce
####recentlyadded


#    delA=np.array([np.sum([wt*force for wt,force in zip(al_wt,distributedF)]) for al_wt in Awt]) #wrong. check lengths
    delA=np.array([np.sum([wt*force for wt,force in zip(al_wt,dcrt_distributedF)]) for al_wt in Awt]) 
    t25=time.time()
    delc=np.sum([wt*f/np.linalg.norm(wt) for wt,f in zip(cwt,distributedF)],0) #I think that normalization also nixes the need to to worry about spacing
    t26=time.time()
    delr0=np.sum(distributedF) + lamR * 2.0 * (r0-r_est)###This line is wrong
    
    t3=time.time()
    
    F_L_a=np.array([2*wl*al for wl,al in zip(Wml,Aml)])
    F_M=2*(Center-c_est) * spacing
    #F_L_b=np.array([2*wl*bl for wl,bl in zip(Wl[1:],Bl)])    
    delA+= lamL * F_L_a        #;delB+= lamL * F_L_b
    delc+=lamM * F_M
    
    t31=time.time()
    
    #F_L_r0=2.0* Wl[0] * (r0-r_est)
    #delr0 += lamL*F_L_r0
    #
    
##new stuff
#    N_approx=10    ###only updates the N_approx biggest frequency changes###
#    N_approx=nF#no change
#    ind=np.argpartition(np.abs(delA),-N_approx)[:-N_approx]
#    delA[ind]=0
##new stuff    
    
    Aml-=tau*delA
#    Bml-=tau*delB*100
    r0-=tau*delr0
    Center -=tau*delc

    
    
    Radius-=tau*delr0 #changes radius for every angle

    for da,b in zip(delA,BasisFunctions):
        if da!=0:
            Radius-=tau*da*b

    
    if True in Radius<0:
        print "Warning: radius negative at iter ",iteration

    t32=time.time()
    t33=time.time()
    t34=time.time()       
    tend=time.time()

#    PhiRecord.append(phi)
    RadiusRecord.append(Radius.copy())
    tmp=np.copy(Center)
    CenterRecord.append(tmp)
    if debug:
        print '\niteration ',iteration    
        print 'r0 is ',r0
        print 'c is ',Center
        print 'normA is ',np.linalg.norm(Aml)
#    print 'Iteration ',iteration,' Total Time ',tend-t0
    print 'It ',iteration,' Center ',Center ,' r0 ',r0 ,' |A| ',np.linalg.norm(Aml)
    print 'Total Time ',tend-t0
    if debug_time:
        print 'Time calculating u0,u1,bdryidx: ', t1-t0
#        print 'Time calculating Iangle: ', t2-t1131
        print 'Time using dels to update vars: ',t32-t31
        print 'Time calculating Dels: ', t3-t2
        print 'Recreating phi for each angle: ',t33-t32
        print 'summing numpy array: ', t34-t33
#        print t01-t0
        if debug_time==2:
            
            print 'looking at first section:'
            print 't2',t02
            print 't03',t03-t02
            print 't04',t04-t03
            print 't05',t05-t04
            print 't06',t06-t05
            print 't07',t07-t06
            print 't08',t08-t07
            print 't09',t09-t08
            print 't010',t010-t09
            print 't011',t011-t010
            print 't012',t012-t011
            
            print 'looking at dels'
            print 't21',t21-t2
            print 't22',t22-t21
            print 't23',t23-t22
            print 't24',t24-t23
            print 't25',t25-t24
            print 't26',t26-t25
            print 't26-t3',t3-t26
#        print t013
#        print t014

#        print t02-t0
#        print t03-t0
#        print t04-t0
#        print t05-t0
#        print t06-t0
#        print t07-t0
#        print t08-t0
#        print t09-t0
#        print t010-t0
#        print t011-t0
#        print t012-t0
#        print t013-t0
#        print t014-t0
#u1pts = np.flatnonzero(phi<=0)                 # interior points
#u0pts = np.flatnonzero(phi>0)                  # exterior points
#u1 = np.sum(I.flat[u1pts])/(len(u1pts)+eps) # interior mean


print '\nMarker ',marker
print 'Interior Intensity: ',u1*Iscale,'\n'










#a=stophere

#for i,ix in enumerate(ref.AngleIdx_Flat):
#    if 11479 in ix:
#        print i
        


#phiR0=Idist-R0
#mlab.contour3d(sIzi,sIxi,sIyi,phiR0,color=(0,1,0),opacity=0.25,contours=[0])

####end loop ######


#a=stop

#from mayavi.filters.delaunay3d import Delaunay3D
from mayavi import mlab
from mayavi.api import Engine
from mayavi.modules.surface import Surface


#Radius=RadiusRecord[0]

#def myscene():

AngleWrap=np.hstack([Angle,Angle[:,0,:][:,np.newaxis,:]])
theta_mesh,phi_mesh=AngleWrap
RadiusWrap=np.vstack([Radius,Radius[0,:][np.newaxis,:]])#Add theta at 2pi in addition to at 0 
mz= RadiusWrap*np.cos(phi_mesh) + Center[0]                    *spacing[0]
mx= RadiusWrap*np.sin(phi_mesh)*np.cos(theta_mesh) + Center[1] *spacing[1]
my= RadiusWrap*np.sin(phi_mesh)*np.sin(theta_mesh) + Center[2] *spacing[2]
#    mz=( RadiusWrap*np.cos(phi_mesh) + Center[0]                    )*spacing[0]
#    mx=( RadiusWrap*np.sin(phi_mesh)*np.cos(theta_mesh) + Center[1] )*spacing[1]
#    my=( RadiusWrap*np.sin(phi_mesh)*np.sin(theta_mesh) + Center[2] )*spacing[2]    



engine = Engine()
engine.start()
if len(engine.scenes) == 0:
    engine.new_scene()

scene = engine.scenes[0]
#scene.scene.disable_render = True
#delaunay3d = Delaunay3D()
#mlab.pipeline.scalar_scatter(mz,mx,my,np.ones(my.shape))
#vtk_data_source = engine.scenes[0].children[0]
#engine.add_filter(delaunay3d, obj=vtk_data_source)
#delaunay3d.name = 'Delaunay3D'
#scene.scene.disable_render = False
#surface = Surface()
#engine.add_filter(surface, delaunay3d)

surface=mlab.mesh(mz,mx,my)
surface.actor.property.opacity = 0.4
surface.actor.actor.origin = np.array([ 0.,  0.,  0.])
surface.actor.actor.scale = np.array([ 1.,  1.,  1.])
surface.actor.actor.orientation = np.array([ 0., -0.,  0.])
surface.actor.actor.estimated_render_time = 2.002716064453125e-05
surface.actor.actor.render_time_multiplier = 0.7761158533253418
surface.actor.actor.reference_count = 3
surface.actor.actor.position = np.array([ 0.,  0.,  0.])
surface.actor.property.specular_color = (1.0, 0.0, 0.0)
surface.actor.property.diffuse_color = (1.0, 0.0, 0.0)
surface.actor.property.ambient_color = (1.0, 0.0, 0.0)
surface.actor.property.color = (1.0, 0.0, 0.0)
surface.actor.mapper.progress = 1.0
surface.actor.mapper.scalar_range = np.array([ 1.,  1.])
surface.actor.mapper.scalar_visibility = False



##Add Image

sIzi,sIxi,sIyi=spacing[0]*Izi,spacing[1]*Ixi,spacing[2]*Iyi


volume=mlab.pipeline.volume(mlab.pipeline.scalar_field(sIzi,sIxi,sIyi,I))

##Add current centroid
mlab.points3d(spacing[0]*Center[0],spacing[1]*Center[1],spacing[2]*Center[2],color=(1,1,1))     
mlab.points3d(spacing[0]*c_est[0],spacing[1]*c_est[1],spacing[2]*c_est[2],color=(0,0,0))




#volume=mlab.pipeline.volume(mlab.pipeline.scalar_field(sIzi,sIxi,sIyi,ref.Dcrt_Angle))
#
#
#from mayavi.sources.array_source import ArraySource
#array_source = ArraySource(scalar_data=I)
#array_source.spacing=spacing
#
#engine.add_source(array_source)
#scene = engine.scenes[0]
#scene.scene.disable_render = True
#from mayavi.modules.volume import Volume
#volume = Volume()

#from tvtk.util.ctf import PiecewiseFunction
#otf = PiecewiseFunction()
#
##otf.add_point(0.0,0.0)
##otf.add_point(0.25,0.0)
##otf.add_point(0.35,0.15)
##otf.add_point(1.0,.25)
#
#
#otf.add_point(0.0,0.0)
#otf.add_point(25,0.0)
#otf.add_point(35,0.15)
#otf.add_point(100,.25)
#
#vp.set_scalar_opacity(otf)
#
#
#
#volume.volume_property.set_scalar_opacity(otf)



#engine.add_module(volume, obj=array_source)
##volume.name = 'Volume'
#volume.update_ctf = True
#
#
#scene.scene.disable_render = False
















#
#scene=engine.new_scene()
#volume=mlab.pipeline.volume(mlab.pipeline.scalar_field(sIzi,sIxi,sIyi,I))
#
#scene.scene.disable_render = True
#from mayavi.modules.volume import Volume
#scalarfield=scene.add_child(mlab.pipeline.scalar_field(sIzi,sIxi,sIyi,I))
#volume=mlab.pipeline.volume(mlab.pipeline.scalar_field(sIzi,sIxi,sIyi,I))
##volume=engine.scenes[-1].children[1].children[0].children[0]
#volume.volume_property.scalar_opacity_unit_distance = 2.0
#
#
#
#volume=mlab.pipeline.volume(mlab.pipeline.scalar_field(sIzi,sIxi,sIyi,I))
#vp=volume.volume_property



#otf=vp.get_scalar_opacity()


#otf.remove_all_points()
#from tvtk.util.ctf import PiecewiseFunction
#otf = PiecewiseFunction()
#otf.add_point(0.0,0.0)
##otf.add_point(0.25,0.0)
#otf.add_point(0.35,0.15)
#otf.add_point(1.0,.25)
#vp.set_scalar_opacity(otf)


#ctf=vp.rgb_transfer_function
#vp.set_color(ctf)







#otf.add_point(0.0,0.0)
#otf.add_point(0.25,0.0)
#otf.add_point(0.35,0.15)
#otf.add_point(1.0,.25)


#engine.new_scene()
#
#mlab.mesh(mz,mx,my)
#mlab.points3d(spacing[0]*Center[0],spacing[1]*Center[1],spacing[2]*Center[2],color=(1,1,1))     

#myscene()



#mlab.points3d(mz[0,0],mx[0,0],my[0,0],color=(0,1,0))

###For seeing Forces surrounding the volume:
#K=np.ones(I.shape)*distributedF.min()*2
#K.flat[idx]=distributedF
#SFK=mlab.pipeline.volume(mlab.pipeline.scalar_field(sIzi,sIxi,sIyi,K))
#obj = mlab.pipeline.image_plane_widget(SFK,plane_orientation='y_axes',slice_index=16)



#,vmin=20,vmax=50

#print 'Initial radii:'
#for n in range(22):
#    print np.mean(RadiusRecord[n])



def flat2triple(ix):
    dims=I.shape
    coord3=ix%dims[2]
    coord2=( (ix-coord3)%(dims[1]*dims[2]) )/dims[2]
    coord1=( ix-coord3 - dims[2]*coord2 )/(dims[1]*dims[2])
    return (coord1,coord2,coord3)
    

#for i,f in enumerate(F):
#    if f>0:
#        print i,' ',flat2triple(idx[i])
       
       
       
#def InspectScene(n):
#    phi=PhiRecord[n]
#    engine.new_scene()
#    mlab.pipeline.volume(mlab.pipeline.scalar_field(sIzi,sIxi,sIyi,I))
#    volume=engine.scenes[-1].children[0].children[0].children[0]
##    volume.volume_mapper.blend_mode = 'maximum_intensity'
#    mlab.contour3d(sIzi,sIxi,sIyi,phi,color=(1,0,0),opacity=0.25,contours=[0])    
#    mlab.points3d(spacing[0]*c_est[0],spacing[1]*c_est[1],spacing[2]*c_est[2])


#mlab.title("sddsf")



#InspectScene(n_iter)
#mlab.text(.05,.02,'150303_CM_Hi-res_20x',width=0.3)
##mlab.text(.05,.02,'150302_CM_Hi-res_40x W5',width=0.3)
#
#mlab.text(.05,.07,'marker '+str(marker),width=0.15)
#mlab.text(.05,.10,'iter '+str(n_iter),width=0.08)
#dst_dir='/home1/03176/csnyder/temp_output'





#nsaves=0
#150303_CM_Hi-res_20x
#mlab.savefig(dst_dir+'/150302_Hi-res_40x W5 m' +str(marker)+' 0'+str(nsaves)+'.png');nsaves+=1






#    F=distributedF    
#    normF=np.sum(F**2)
#    Basis=[Y(w,Iangle) for aml,w in zip(Aml,Freq)]  
#Basis=[Y(w,Iangle) for w in Freq]  
#conv=[np.sum(b*distributedF) for b in Basis]


    #############slow as f***
#    u0=0; cnt_outside=0
#    u1=0; cnt_inside=0
#    bdry_idx=[]; bdry_th=[]; bdry_ph=[]
#    bdry_flat_idx=[];
#
#    ####Assign locations####(inside,outside,boundary)
#
##    ref_x,ref_y,ref_z = Ixi-Center[1]+ref.origin[1],Iyi-Center[2]+ref.origin[2],Izi-Center[0]+ref.origin[0]
#
##    for i in xrange(I.size):
##    for x,y,z in zip(Ixi.flat,Iyi.flat,Izi.flat):#xyz tuple
#    for i,(x,y,z) in enumerate(zip(Ixi.flat,Iyi.flat,Izi.flat)):#xyz tuple
#        ref_x,ref_y,ref_z = ref.toReferenceCoordinates(x,y,z,Center)
#        rad=Radius[mapTheta[ref_x,ref_y,ref_z],mapPhi[ref_x,ref_y,ref_z]]
#        dist=Distance[x,y,z]        
#        if dist-rad<=0:
#            u1+=I[z,x,y]
#            cnt_inside+=1
#        elif dist-rad>0:
#            u0+=I[z,x,y]
#            cnt_outside+=1
#        if (dist-rad)<=delta and -delta<=(dist-rad):
#            bdry_idx.append((x,y,z))
#            bdry_th.append(mapTheta[ref_x,ref_y,ref_z])
#            bdry_ph.append(mapPhi[ref_x,ref_y,ref_z])    
#            bdry_flat_idx.append(i)
#    bdry_th=np.array(bdry_th)
#    bdry_ph=np.array(bdry_ph)
#    Iangle=(bdry_th,bdry_ph)
#    idx=np.array(bdry_flat_idx)
#    u0/=cnt_outside
#    u1/=cnt_inside


####
#cy_iteration(
#    Radius,I,
#    std_size,ref.ref_size,
#    ref.ref_shape.astype(np.int),std_shape.astype(np.int),
#    ref.Dcrt_Angle,
#    ref.Distance,
#    Awt,np.array(Awt.shape),
#    Aml,Wml,
#    Center,c_est,
#    r0,r_est,
#    ref.origin,
#    lamP,lamM,lamR,lamL,
#    delta,tau,
#    n_threads,
#    cy_u1,
#    cy_delA,
#    cy_delc,
#    cy_delr0  
#    )