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


from polar_nucleus3D_Cython.cy_sphere_iter import cy_iteration
from polar_nucleus3D_Cython.cy_sphere_iter import cy_get_seg

#tbeginning=time.time()

from scipy.special import sph_harm



#t01=0
#t02=0
#t03=0
#t04=0
#t05=0
#t06=0
#t07=0
#t08=0
#t09=0
#t010=0
#t011=0
#t012=0
#t013=0
#t014=0
def get_spectrum(N):
    Freq=[]
    for l in range(1,N+1):#l=0 could be used as radius
        for m in range(-l,l+1):
            Freq.append( (m,l) )
    nF=len(Freq)
    return nF,Freq

class SphereParams(object):

    theta_steps=40
    phi_steps=20

#        self.N=8 #number of nonzero frequencies
    N=5
    
    #tau=0.01
    tau=0.001
    n_iter=250

    r0=8.0 #initial radius        
    r_est=8.0#needs lamL,Wl[0] nonzero to have effect
#    r0=4.0 #initial radius        
#    r_est=4.0#needs lamL,Wl[0] nonzero to have effect


#        self.r_est=8#needs lamL,Wl[0] nonzero to have effect

    lamP=1.0 
            ###If lamG is too big, will have phi.min() > 0 following level set shrink
            #seems to be working well between 0.01 and 2
    lamG=0.0#Not being used
    lamL=4.5#L2 norm penalty        
    lamM=1.0
    lamR=8.0
    
    #Wl[0]=0#Don't control r0
    
#        self.debug=1        
#        self.delta=3.5#originally 1.2    
#        self.delta=0.7#originally 1.2    
    delta=1.2#originally 1.2

    debug=0
    debug_membership=0
    debug_time=0
    
    n_threads=8
#    n_threads=1

    def __init__(self):
        self.nF,self.Freq=get_spectrum(self.N)
        self.Wml=np.ones(self.nF)
        

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
    def __init__(self,Ishape,spacing):
        self.spacing=spacing
#        self.I=I
        self.Ishape=Ishape
#        self.ref_image=np.zeros(2*np.array(I.shape)+1)
        self.ref_image=np.zeros(np.array(self.Ishape))#even for now... bleh
        self.origin=np.array(self.ref_image.shape)//2
        self.z,self.x,self.y=np.indices(self.ref_image.shape)-self.origin[:,np.newaxis,np.newaxis,np.newaxis]
        self.d,self.th,self.ph=cart2polar(self.x,self.y,self.z,spacing)
        self.Distance=np.sqrt( (self.z*spacing[0])**2 + ((self.x)*spacing[1])**2  +  ((self.y)*spacing[2])**2  )
        self.ref_size=self.ref_image.size
        self.ref_shape=np.array(self.ref_image.shape).astype(np.int)
        
    def set_discrete_angles(self,theta_steps,phi_steps):
        self.d,self.di_th,self.di_ph=cart2_Discrete_polar(self.x,self.y,self.z,self.spacing,theta_steps,phi_steps)
        self.AngleIdx_Flat=[]
        self.AngleIdx=[]
        for t,p in product(range(theta_steps),range(phi_steps)):
            self.AngleIdx_Flat.append( np.flatnonzero( np.logical_and( self.di_th == t, self.di_ph == p) ) )
            self.AngleIdx.append(         np.nonzero( np.logical_and( self.di_th == t, self.di_ph == p) ) )
        self.dist_at_angle=[self.Distance.flat[anxf] for anxf in self.AngleIdx_Flat]

        self.Dcrt_Angle=-1*np.ones(self.ref_image.shape)
        for a in xrange(len(self.AngleIdx_Flat)):
            self.Dcrt_Angle.flat[self.AngleIdx_Flat[a]]=a
        self.Dcrt_Angle=self.Dcrt_Angle.astype(np.int)
        self.Dcrt_Angle=np.ascontiguousarray(self.Dcrt_Angle,dtype=np.int)###Added 2015-8-26


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




#def shift_idx_ref2std_flat(idx,Center):
class PolarCache(SphereParams):
    
    def __init__(self,patch_shape,spacing):
        self.spacing=spacing
        SphereParams.__init__(self)
        self.patch_shape=patch_shape
        
        theta_spacing = 2*np.pi/self.theta_steps
#        phi_spacing=np.pi/(self.phi_steps-1)
        self.theta_grid=np.mgrid[-np.pi:np.pi:theta_spacing]
        self.phi_grid=np.mgrid[0:np.pi:np.complex(0,self.phi_steps)]
    
    
        #theta_grid,phi_grid=np.mgrid[0:2*np.pi:theta_spacing,0:np.pi:np.complex(0,phi_steps)]
        
        self.Angle=np.mgrid[-np.pi:np.pi:theta_spacing,0:np.pi:np.complex(0,self.phi_steps)]
        
        self.BasisFunctions=[Y(ml,self.Angle) for ml in self.Freq]
        
        self.Aml=np.zeros(self.nF)
#        self.Wml=np.ones(self.Aml.shape).astype(np.float)

        self.ref=ReferenceGrid(self.patch_shape,self.spacing)  #Do this later. Hope radius doesn't hit boundary
        self.ref.set_discrete_angles(self.theta_steps,self.phi_steps)
        self.ref_dims=np.array(self.ref.ref_image.shape)
        self.std_dims=np.array(self.patch_shape)
        self.mapTheta,self.mapPhi=self.ref.getCorrespondence(self.Angle)#z,x,y #tricky 

        Itheta=[];Iphi=[]    #ugly patch to get angles
        for dcrt_theta,dcrt_phi in self.AngleIterator():     
            Itheta.append( self.theta_grid[dcrt_theta] )
            Iphi.append( self.phi_grid[dcrt_phi])
        Itheta=np.array(Itheta)
        Iphi=np.array(Iphi)
        Iangle=(Itheta,Iphi)
        self.Awt=np.array([Y(w, Iangle ) for w in self.Freq])#can be moved out of loop    

        AngleWrap=np.hstack([self.Angle,self.Angle[:,0,:][:,np.newaxis,:]])
        self.theta_mesh,self.phi_mesh=AngleWrap


    def AngleIterator(self):
        return product(range(self.theta_steps),range(self.phi_steps))    
#class PolarContour(SphereParams,PolarCache):
    

class PolarMesh(SphereParams):

    def __getattr__(self,attr):
        #Did this as a hack to have some of the value of PolarMesh precomputed as they should be the same for every mesh.
        return getattr(self.polar_cache,attr)#If it's not defined here, find it by searching in the "polar_cache" member

    def __init__(self,polar_cache,I,Iscale,spacing,c_est):
        SphereParams.__init__(self)        
        self.polar_cache=polar_cache        
        self.I=I
        self.Iscale=Iscale        
        
        if not (np.array(self.patch_shape)==np.array(self.I.shape)).all():
            print 'warning: cache and mesh window sizes are different'
            print 'cache shape (patch_shape):',self.patch_shape
            print 'window size:',I.shape
        
        
        self.Radius=np.zeros(self.Angle[0].shape)+self.r0 #theta_steps x phi_steps 

        
        self.I=np.ascontiguousarray(I)##CRITICAL if patch is being sliced from a larger WormImage
        self.Iscale=Iscale
#        self.Iscale=raw_image.max()
#        float_image=raw_image.astype(np.float64)
#        self.I=float_image/float_image.max()
        self.Izi,self.Ixi,self.Iyi=np.indices(self.I.shape)        
        #    I=I.astype(np.float64) #moved to inside PolarContour class
#    Iscale=I.max()
#    I/=Iscale
        
                
        
        self.c_est=c_est
        self.Center=np.array(c_est)[:]
        self.spacing=spacing        
        self.interior_intensity=0#placeholder
        
        
        self.CenterRecord=[]
        self.CenterRecord.append(self.Center.copy())
        self.RadiusRecord=[]
        self.RadiusRecord.append(self.Radius.copy())
        
#        self.Freq=[]
#        for l in range(1,self.N+1):#l=0 could be used as radius
#            for m in range(-l,l+1):
#                self.Freq.append( (m,l) )
#        self.nF=len(self.Freq)        
        #Theta 0,2pi    phi 0,pi                
        if self.n_iter==1:
            self.debug_membership=1
        self.iteration=0      
        self.std_size=self.I.size
        self.std_shape=np.array(self.I.shape).astype(np.int_)



    def ref3d_2_stdflat(self,idx,Center):
        if not idx[0].size:
            return np.empty(0)
        Center=np.round(Center)###May change later to allow more resolution
        v_displace =  np.round(Center) - self.ref.origin #z,x,y
        new_idx=[ ix+vd for ix,vd in zip(idx,v_displace) ]#for each dim
        valid_ones=np.logical_and.reduce((new_idx[0]>=0,new_idx[0]<self.std_dims[0],new_idx[1]>=0,new_idx[1]<self.std_dims[1],new_idx[2]>=0,new_idx[2]<self.std_dims[2]))
        if True in valid_ones:
            final_idx=[nx[valid_ones] for nx in new_idx]#for each dim
            return np.array( [n0*self.std_dims[1]*self.std_dims[2] + n1*self.std_dims[2] + n2   for n0,n1,n2 in zip(final_idx[0],final_idx[1],final_idx[2]) ]).astype('int')#convert to flat
        else:
            print "Warning: Center has moved far to the side. no corresponding indicies in std system from ref system"
            return np.empty(0)





#tminus1=time.time()
#print 'Time till start of loop: ',tminus1-tbeginning




#for iteration in range(n_iter):


    def do_iter(self):
        self.iteration+=1
#        t0=time.time()    
        print 'Iteration ',self.iteration
        centered_Izi=self.Izi-self.Center[0];  centered_Ixi=self.Ixi-self.Center[1];   centered_Iyi=self.Iyi-self.Center[2]    
    #    Distance=np.sqrt( (centered_Izi*spacing[0])**2 + ((centered_Ixi)*spacing[1])**2  +  ((centered_Iyi)*spacing[2])**2  )    
#        t_areyoukiddingme=time.time()
    ####For each discrete angle, grab stuff of the appropriate bleh####
#        BoundaryPointsPerAngle=[]
#        u0=0.05; #cnt_outside=0
        self.u1=0; cnt_inside=0
        idx_at_angle=[]
        sum_bdry_intensity_at_angle=[]
        n_points_on_bdry_at_angle=[]#maybe do something with this later to adjust for nonuniform boundary segment
        for i,(dcrt_theta,dcrt_phi) in enumerate(self.AngleIterator()):##Iterate over every slice of angle
            
#            t01=time.time()
#            t02+=time.time()-t01
            R=self.Radius[dcrt_theta,dcrt_phi]
#            t03+=time.time()-t01
    #        outside_idx =[ ix[ R < ref.Distance.flat[ref.AngleIdx_Flat[i]] ] for ix in ref.AngleIdx[i] ]
#            t04+=time.time()-t01
            #Let's just skip this section for speed: Call it u0=0.05
    #        std_flat_outside_idx= ref3d_2_stdflat( outside_idx, Center)
    #        if std_flat_outside_idx.size:#checks if not empty
    #            u0+=np.sum(I.flat[std_flat_outside_idx])
    #            cnt_outside+=len(std_flat_outside_idx)
    #        else:
    #            if debug_membership:
    #                print 'angle index ',i,' has empty outside at d_th=',dcrt_theta,' d_ph=',dcrt_phi
#            t05+=time.time()-t01
    #        inside_idx  =[ ix[ R > ref.Distance.flat[ref.AngleIdx_Flat[i]] ] for ix in ref.AngleIdx[i] ]    #Costly
            inside_idx  =[ ix[ R > self.ref.dist_at_angle[i] ] for ix in self.ref.AngleIdx[i] ]    #Costly
#            t06+=time.time()-t01
            std_flat_inside_idx = self.ref3d_2_stdflat(  inside_idx, self.Center)    #Costly
#            t07+=time.time()-t01
            if std_flat_inside_idx.size:
                self.u1+=np.sum(self.I.flat[std_flat_inside_idx])
                cnt_inside+=len(std_flat_inside_idx)
            else:
                if self.debug_membership:
                    print 'angle index ',i,' has empty inside at   d_th=',dcrt_theta,' d_ph=',dcrt_phi
#            t08+=time.time()-t01
            #Costly next line
    #        bdry_idx=[ ix[np.logical_and( R-ref.Distance.flat[ref.AngleIdx_Flat[i]] <= delta , ref.Distance.flat[ref.AngleIdx_Flat[i]] - R <= delta)]   for ix in ref.AngleIdx[i] ]
            bdry_idx=[ ix[np.logical_and( R-self.ref.dist_at_angle[i] <= self.delta , self.ref.dist_at_angle[i] - R <= self.delta)]   for ix in self.ref.AngleIdx[i] ]
#            t09+=time.time()-t01
            std_flat_bdry_idx=self.ref3d_2_stdflat( bdry_idx, self.Center)
#            t010+=time.time()-t01
            idx_at_angle.append(std_flat_bdry_idx)
#            t011+=time.time()-t01
            if std_flat_bdry_idx.size:
                sum_bdry_intensity_at_angle.append(np.sum(self.I.flat[std_flat_bdry_idx]))
                n_points_on_bdry_at_angle.append(len(std_flat_bdry_idx))
            else:
                n_points_on_bdry_at_angle.append(0)
                if self.debug_membership:
                    print 'angle index ',i,' has empty boundary at d_th=',dcrt_theta,' d_ph=',dcrt_phi
    #        if std_flat_bdry_idx.size:    
    #            break
#            t012+=time.time()-t01
        self.u1/=cnt_inside
        self.interior_intensity=self.u1*self.Iscale
    #    u0/=cnt_outside ######skipped for speed!
        self.u0=0.05
    #        a=stoooooop_at_first_iter______________________________
    #    t013=time.time()#-t012      
    #    t014=time.time()#-t014  
        idx=list(chain(*idx_at_angle))#conforms to old notation
        dcrt_I=np.zeros(len(idx_at_angle))
        for ang,ix in enumerate(idx_at_angle):
            if ix.size:
                dcrt_I[ang]=np.mean(self.I.flat[ix])
#        t1=time.time()
    #    a=stoooooop
        #Nothing needed to do here
#        t2=time.time()
        cwt=np.hstack([centered_Izi.flat[idx][:,np.newaxis], centered_Ixi.flat[idx][:,np.newaxis], centered_Iyi.flat[idx][:,np.newaxis] ])
        F_P=np.array([ (self.I.flat[ix]-self.u1)**2-(self.I.flat[ix]-self.u0)**2   for ix in idx])#wrong
        distributedF = self.lamP*F_P
    #    dcrt_F_P=np.array([ (I.flat[ix]-u1)**2-(I.flat[ix]-u0)**2   for ix in idx])
    #    dcrt_F_P= (dcrt_I-u1)**2  - (dcrt_I-u0)**2 #Maybe not fully equivalent but close
    #    dcrt_F_P=  [np.sum((I.flat[ix]-u1)**2-(I.flat[ix]-u0)**2)   for ix in idx_at_angle]#Each ix here is an array of pix at that angle
        dcrt_F_P=np.zeros(len(idx_at_angle))#loop is eq to above
        for a,ix in enumerate(idx_at_angle):
            if ix.size:
                dcrt_F_P[a]=np.sum((self.I.flat[ix]-self.u1)**2-(self.I.flat[ix]-self.u0)**2)
        dcrt_distributedF= self.lamP*dcrt_F_P
    ####recentlyadded
    #    scaleforce=spacing[0]-np.sin(Ipsi.flat[idx])
    #    distributedF*=scaleforce
    ####recentlyadded
    #    delA=np.array([np.sum([wt*force for wt,force in zip(al_wt,distributedF)]) for al_wt in Awt]) #wrong. check lengths
        delA=np.array([np.sum([wt*force for wt,force in zip(al_wt,dcrt_distributedF)]) for al_wt in self.Awt]) 
        delc=np.sum([wt*f/np.linalg.norm(wt) for wt,f in zip(cwt,distributedF)],0) #I think that normalization also nixes the need to to worry about spacing
        delr0=np.sum(distributedF) + self.lamR * 2.0 * (self.r0-self.r_est)###This line is wrong
#        t3=time.time()
        F_L_a=np.array([2*wl*al for wl,al in zip(self.Wml,self.Aml)])   
        F_M=2*(self.Center-self.c_est) * self.spacing
        #F_L_b=np.array([2*wl*bl for wl,bl in zip(Wl[1:],Bl)])    
        delA+= self.lamL * F_L_a        #;delB+= lamL * F_L_b
        delc+=self.lamM * F_M
        
        #Comparison of forces acting on center:
        print 'edge forces:',delc
        print 'center force:',F_M
#        t31=time.time()
        #F_L_r0=2.0* Wl[0] * (r0-r_est)
        #delr0 += lamL*F_L_r0
        #
    ##new stuff
    #    N_approx=10    ###only updates the N_approx biggest frequency changes###
    #    N_approx=nF#no change
    #    ind=np.argpartition(np.abs(delA),-N_approx)[:-N_approx]
    #    delA[ind]=0
    ##new stuff    
        self.Aml-=self.tau*delA
    #    Bml-=tau*delB*100
        self.r0-=self.tau*delr0
        self.Center -=self.tau*delc
        self.Radius-=self.tau*delr0 #changes radius for every angle
        for da,b in zip(delA,self.BasisFunctions):
            if da!=0:
                self.Radius-=self.tau*da*b
        if True in self.Radius<0:
            print "Warning: radius negative at iter ",self.iteration
#        t32=time.time()
#        t33=time.time()
#        t34=time.time()       
#        tend=time.time()
    #    PhiRecord.append(phi)
        self.RadiusRecord.append(self.Radius.copy())
        self.CenterRecord.append(self.Center.copy())
        if self.debug:
            print '\niteration ',self.iteration    
            print 'r0 is ',self.r0
            print 'c is ',self.Center
            print 'normA is ',np.linalg.norm(self.Aml)
    #    print 'Iteration ',iteration,' Total Time ',tend-t0
        print 'It ',self.iteration,' Center ',self.Center ,' r0 ',self.r0 ,' |A| ',np.linalg.norm(self.Aml)
#        print 'Total Time ',tend-self.t0
#        if self.debug_time:
#            print 'Time calculating u0,u1,bdryidx: ', t1-t0
#            print 'Time calculating Iangle: ', t2-t1
#            print 'Time calculating Dels: ', t3-t2
#            print 'Recreating phi for each angle: ',t33-t32
#            print 'summing numpy array: ', t34-t33
#    #        print t01-t0
#            if self.debug_time==2:
#                print 'looking at first section:'
#                print 't2',t02
#                print 't03',t03-t02
#                print 't04',t04-t03
#                print 't05',t05-t04
#                print 't06',t06-t05
#                print 't07',t07-t06
#                print 't08',t08-t07
#                print 't09',t09-t08
#                print 't010',t010-t09
#                print 't011',t011-t010
#                print 't012',t012-t011
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

    def do_cython_iter(self):
        self.iteration+=1        
        
#        print "self.lamL is, ",self.lamL
#        print 'self.Wml5 is ',self.Wml[5]
#        print 'obj_polar562, lamM is ',self.lamM

#        cy_inputs=[self.Radius,self.I,self.std_size,self.ref.ref_size,self.ref.ref_shape.astype(np.int),
#                   self.std_shape.astype(np.int),self.ref.Dcrt_Angle,self.ref.Distance,self.Awt,np.array(self.Awt.shape),
#            self.Aml,self.Wml,self.Center,self.c_est,self.r0,self.r_est,self.ref.origin,
#            self.lamP,self.lamM,self.lamR,self.lamL,self.delta,self.tau,self.n_threads]     
##        if self.iteration==0:#debug
#        for cy in cy_inputs:
#            try:
#                print cy.flags['C_CONTIGUOUS']
#            except:
#                print 'not a np array'
                
       
                
        
        cy_u1,cy_delA,cy_delc,cy_delr0=cy_iteration(
            self.Radius,self.I,self.std_size,self.ref.ref_size,self.ref.ref_shape.astype(np.int),
            self.std_shape.astype(np.int),self.ref.Dcrt_Angle,self.ref.Distance,self.Awt,np.array(self.Awt.shape),
            self.Aml,self.Wml,self.Center,self.c_est,self.r0,self.r_est,self.ref.origin,
            self.lamP,self.lamM,self.lamR,self.lamL,self.delta,self.tau,self.n_threads
            )
            

        self.Aml    -= self.tau* cy_delA
        self.r0     -= self.tau* cy_delr0
        self.Center -= self.tau* cy_delc

        self.u1=cy_u1
        
        self.interior_intensity=self.u1*self.Iscale
        
        self.Radius-=self.tau* cy_delr0 #changes radius for every angle
    
        for da,b in zip(cy_delA,self.BasisFunctions):
            if da!=0:#does this actually help the speedup??--yes only when we allow just a few coords to update
                self.Radius-=self.tau*da*b
    
        
        if True in self.Radius<0:
            print "Warning: radius negative at iter ",self.iteration
        

    def return_segmented_image(self):
        self.seg_patch=cy_get_seg(
            self.Radius,self.I,self.std_size,self.ref.ref_size,self.ref.ref_shape.astype(np.int),
            self.std_shape.astype(np.int),self.ref.Dcrt_Angle,self.ref.Distance,self.Center,self.ref.origin,
            self.n_threads
            )
        if not ( np.unique(self.seg_patch)==np.array([0,1]) ).all():
            print 'Shouldnt happen: segmented array should be all 0 or 1'
        self.seg_patch=self.seg_patch.astype(np.bool)
        return self.seg_patch
    

    
    
if __name__=='__main__':
    
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
    
    
    
#    I=I.astype(np.float64) #moved to inside PolarContour class
#    Iscale=I.max()
#    I/=Iscale
    
    
    #spacing=np.array([3.0,1.0,1.0])#300x100x100um #150226 Hi-res_40x
    #spacing=np.array([2.0,1.0,1.0])#300x100x100um #150226 Hi-res_40x
    spacing=np.array([1.0,1.0,1.0])
    
    
    c_est=0.5*np.array(I.shape).astype(np.float)#z,x,y

#    spherule=PolarContour(I,spacing,c_est)
#    spherule.do_iter()


































#    spherule.do_iter()


#    print '\nMarker ',marker
#    print 'Interior Intensity: ',u1*Iscale,'\n'

    
#    from mayavi import mlab
#    from mayavi.api import Engine
#    from mayavi.modules.surface import Surface
##    
#    
#    Radius=spherule.RadiusRecord[0]
#    
#    #def myscene():
#    
#    AngleWrap=np.hstack([Angle,Angle[:,0,:][:,np.newaxis,:]])
#    theta_mesh,phi_mesh=AngleWrap
#    RadiusWrap=np.vstack([Radius,Radius[0,:][np.newaxis,:]])#Add theta at 2pi in addition to at 0 
#    mz= RadiusWrap*np.cos(phi_mesh) + Center[0]                    *spacing[0]
#    mx= RadiusWrap*np.sin(phi_mesh)*np.cos(theta_mesh) + Center[1] *spacing[1]
#    my= RadiusWrap*np.sin(phi_mesh)*np.sin(theta_mesh) + Center[2] *spacing[2]
#    #    mz=( RadiusWrap*np.cos(phi_mesh) + Center[0]                    )*spacing[0]
#    #    mx=( RadiusWrap*np.sin(phi_mesh)*np.cos(theta_mesh) + Center[1] )*spacing[1]
#    #    my=( RadiusWrap*np.sin(phi_mesh)*np.sin(theta_mesh) + Center[2] )*spacing[2]    
#    
#    
#    
#    engine = Engine()
#    engine.start()
#    if len(engine.scenes) == 0:
#        engine.new_scene()
#    
#    scene = engine.scenes[0]
#    #scene.scene.disable_render = True
#    #delaunay3d = Delaunay3D()
#    #mlab.pipeline.scalar_scatter(mz,mx,my,np.ones(my.shape))
#    #vtk_data_source = engine.scenes[0].children[0]
#    #engine.add_filter(delaunay3d, obj=vtk_data_source)
#    #delaunay3d.name = 'Delaunay3D'
#    #scene.scene.disable_render = False
#    #surface = Surface()
#    #engine.add_filter(surface, delaunay3d)
#    
#    surface=mlab.mesh(mz,mx,my)
#    surface.actor.property.opacity = 0.4
#    surface.actor.actor.origin = np.array([ 0.,  0.,  0.])
#    surface.actor.actor.scale = np.array([ 1.,  1.,  1.])
#    surface.actor.actor.orientation = np.array([ 0., -0.,  0.])
#    surface.actor.actor.estimated_render_time = 2.002716064453125e-05
#    surface.actor.actor.render_time_multiplier = 0.7761158533253418
#    surface.actor.actor.reference_count = 3
#    surface.actor.actor.position = np.array([ 0.,  0.,  0.])
#    surface.actor.property.specular_color = (1.0, 0.0, 0.0)
#    surface.actor.property.diffuse_color = (1.0, 0.0, 0.0)
#    surface.actor.property.ambient_color = (1.0, 0.0, 0.0)
#    surface.actor.property.color = (1.0, 0.0, 0.0)
#    surface.actor.mapper.progress = 1.0
#    surface.actor.mapper.scalar_range = np.array([ 1.,  1.])
#    surface.actor.mapper.scalar_visibility = False
#    
#    
#    
#    ##Add Image
#    
#    sIzi,sIxi,sIyi=spacing[0]*Izi,spacing[1]*Ixi,spacing[2]*Iyi
#    
#    
#    volume=mlab.pipeline.volume(mlab.pipeline.scalar_field(sIzi,sIxi,sIyi,I))
#    
#    
#    
#    
#    
#    from mayavi.sources.array_source import ArraySource
#    array_source = ArraySource(scalar_data=I)
#    array_source.spacing=spacing
#    
#    engine.add_source(array_source)
#    scene = engine.scenes[0]
#    scene.scene.disable_render = True
#    from mayavi.modules.volume import Volume
#    volume = Volume()
#    
#    from tvtk.util.ctf import PiecewiseFunction
#    otf = PiecewiseFunction()
#    
#    #otf.add_point(0.0,0.0)
#    #otf.add_point(0.25,0.0)
#    #otf.add_point(0.35,0.15)
#    #otf.add_point(1.0,.25)
#    
#    
#    otf.add_point(0.0,0.0)
#    otf.add_point(25,0.0)
#    otf.add_point(35,0.15)
#    otf.add_point(100,.25)
#    
#    vp.set_scalar_opacity(otf)
#    
#        
#    volume.volume_property.set_scalar_opacity(otf)
#    
#    
#    
#    engine.add_module(volume, obj=array_source)
#    #volume.name = 'Volume'
#    volume.update_ctf = True
#    
#    
#    scene.scene.disable_render = False
#    
#    
#    
# 
#    
#    scene=engine.new_scene()
#    volume=mlab.pipeline.volume(mlab.pipeline.scalar_field(sIzi,sIxi,sIyi,I))
#    
#    scene.scene.disable_render = True
#    from mayavi.modules.volume import Volume
#    scalarfield=scene.add_child(mlab.pipeline.scalar_field(sIzi,sIxi,sIyi,I))
#    volume=mlab.pipeline.volume(mlab.pipeline.scalar_field(sIzi,sIxi,sIyi,I))
#    #volume=engine.scenes[-1].children[1].children[0].children[0]
#    volume.volume_property.scalar_opacity_unit_distance = 2.0
#    
#    
#    
#    volume=mlab.pipeline.volume(mlab.pipeline.scalar_field(sIzi,sIxi,sIyi,I))
#    vp=volume.volume_property
#    #otf=vp.get_scalar_opacity()
#    
#    
#    #otf.remove_all_points()
#    from tvtk.util.ctf import PiecewiseFunction
#    otf = PiecewiseFunction()
#    otf.add_point(0.0,0.0)
#    #otf.add_point(0.25,0.0)
#    otf.add_point(0.35,0.15)
#    otf.add_point(1.0,.25)
#    
#    vp.set_scalar_opacity(otf)
#    
#    
#    #ctf=vp.rgb_transfer_function
#    #vp.set_color(ctf)
#    
#    
#    ##Add current centroid
#    mlab.points3d(spacing[0]*Center[0],spacing[1]*Center[1],spacing[2]*Center[2],color=(1,1,1))     
#    mlab.points3d(spacing[0]*c_est[0],spacing[1]*c_est[1],spacing[2]*c_est[2],color=(0,0,0))



    ##############################
    





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