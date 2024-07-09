# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 09:19:21 2015

@author: cgs567
"""

#from WormBox.ioworm import ls,return_those_ending_in
#from WormBox.ioworm import getPIMWormData, sort_nicely
#from WormBox.WormPlot import squarebysideplot,sidebysideplot,getsquarebysideaxes
#import matplotlib.pyplot as plt
import numpy as np
import skimage.io as skio
from WormBox.WormFormat import autoFijiStyleScaling
try:
    from skimage.filters import threshold_otsu,gaussian_filter
except: #is python3
    from skimage.filter import threshold_otsu,gaussian_filter
#import skimage.measure as m #for connected components
#from itertools import cycle
#from skimage.filters import sobel_h, sobel_v, sobel
#from scipy.linalg import circulant
#from numpy.linalg import solve
import time
from tifffile import imsave

from WormBox.WormFuns import npinfo #asks for len list, shape nparray within, etc when applicabale
from skimage.transform import rotate

#from joblib import delayed

from WormBox.BaseClasses import Worm,Environment

import scipy.ndimage as ndimage

from scipy.special import sph_harm

from multiprocessing import Pool as ThreadPool
from multiprocessing import freeze_support
#import multiprocessing as mp

#from functools import partial
from itertools import product

import time


class SphereParams(object):
    def __init__(self):
#        self.N=8 #number of nonzero frequencies
        self.N=5
        
        self.r0_est=8.0
        self.r0=8.0 #initial radius        
        #tau=0.01
        self.tau=0.001
        self.n_iter=1
#        self.n_iter=4
        self.r_est=12#needs lamL,Wl[0] nonzero to have effect
#        self.c_est=np.array([37,37])#needs lamM nonzero to have effect
        self.lamP=1.0        
                ###If lamG is too big, will have phi.min() > 0 following level set shrink
                #seems to be working well between 0.01 and 2
        self.lamG=0.0
        
        self.lamL=5.0#L2 norm penalty
        
        self.lamM=0.0#not  yet implemented
        
        self.lamR=0.5
        
        #Not currently used : Wl
        self.Wl=np.ones(self.N+1)#include 0 position   ####Need change for this problem
        
        #Wl[0]=0#Don't control r0
        
        self.debug=1
        
#        self.delta=3.5#originally 1.2    
#        self.delta=0.7#originally 1.2    
        self.delta=1.2#originally 1.2    
    
        self.n_jobs=8


def Y(freq,angle):
    m,n=freq
    theta,psi=angle
    if m<0:
        ans= np.sqrt(2)*( -1)**m * np.imag(sph_harm(-m,n,theta,psi))
    elif m==0:
        ans= np.real(sph_harm(m,n,theta,psi))
    elif m>0:
        ans= np.sqrt(2)*( -1)**m * np.real(sph_harm(m,n,theta,psi))
    return np.sum(ans)


#self.it=product( zip(self.Aml,self.Freq) ,parIangle_    )  #This is the correct order!
def parY(my_tuple):
#    print 'tup',my_tuple
    tupl2,angle=my_tuple
    aml,w=tupl2
    return aml*Y(w,angle)

def mask2phi(mask,spacing):
    dt2=ndimage.distance_transform_bf(mask,'euclidean',sampling=spacing)
    dt1=ndimage.distance_transform_bf(1.0-mask,'euclidean',sampling=spacing)
    return dt1-dt2+mask.astype('float64') - 0.5

def getbdry(phi,delta):
    return np.flatnonzero( np.logical_and( phi <= delta, phi >= -delta) )

def cart2polar(x,y,z,spacing):
    z*=spacing[0]
    y*=spacing[2]
    x*=spacing[1]
    theta = np.arctan2(y,x)
    xy=np.sqrt(x**2 + y**2)
    d=np.sqrt(x**2 + y**2 + z**2)
    psi=np.arctan2(xy,z)
    return d,theta,psi





def dummy(tup):
    return np.pi









class Shape(SphereParams):
    def __init__(self,arry64,spacing):
        self.spacing=spacing
        SphereParams.__init__(self)
        self.I=arry64
        

        self.__init_params__()
        self.__init_phi__()
        
    def __init_phi__(self):
        Loc=np.vstack([g.ravel() for g in np.indices(self.I.shape)])        
        distance=np.power(np.sum( np.power(  (Loc-self.c_est[:,np.newaxis])*spacing[:,np.newaxis]  ,2),axis=0),0.5)
        Zero=np.zeros(I.shape)
        mask=np.copy(Zero)
        mask.flat[distance<=self.r0]=1.0
        self.phi=mask2phi(mask,sampling=self.spacing)

    def __init_params__(self):
        self.Izi,self.Ixi,self.Iyi=np.indices(self.I.shape)
        self.c_est=0.5*np.array(self.I.shape).astype(np.float)
        self.c=self.c_est.copy()
        self.eps = np.finfo(np.float).eps
        self.Freq=[]
        for l in range(1,self.N+1):#l=0 could be used as radius
            for m in range(-l,l+1):
                self.Freq.append( (m,l) )
        self.nF=len(self.Freq)
        self.Aml=np.zeros(self.nF)
        self.Wml=np.ones(self.nF)
        self.iteration=0
        self.PhiRecord=[]
        
    def do_iter(self):
        start=time.time()
        self.iteration+=1
        print 'iteration is ',self.iteration
        idx=getbdry(self.phi,self.delta)
        u1pts = np.flatnonzero(self.phi<=0)         # interior points
        u0pts = np.flatnonzero(self.phi>0)          # exterior points
        self.u1 = np.sum(I.flat[u1pts])/(len(u1pts)+self.eps) # interior mean
        self.u0 = np.sum(I.flat[u0pts])/(len(u0pts)+self.eps) # exterior mean

    
        cz,cx,cy=self.c
        x,y,z=self.Ixi-cx,self.Iyi-cy,self.Izi-cz
        Idist,Itheta,Ipsi=cart2polar(x,y,z,spacing=self.spacing)
        Iangle=(Itheta.flat[idx],Ipsi.flat[idx])
        Iangle_=(Itheta,Ipsi)

        F_P=np.array([ (self.I.flat[ix]-self.u1)**2-(self.I.flat[ix]-self.u0)**2   for ix in idx])
        distributedF = self.lamP*F_P
        cwt=np.hstack([z.flat[idx][:,np.newaxis], x.flat[idx][:,np.newaxis], y.flat[idx][:,np.newaxis]])

        
        ######Make sure you update delr0 accounting for spacing difference when you get here
        
###$#$$$

#        print 'dF is ',dF

#        if self.debug==1:
#            print '\niteration ',self.iteration    
##            print 'u0 is ',self.u0,' u1 is ',self.u1
##            print 'len(idx) is ',len(idx)    
##            print 'delr0 is ',delr0
##            print 'delc is ', delc
#            print 'r0 is ',r0
#            print 'c is ',c
#            print 'normA is ',np.linalg.norm(Aml)
            
        ##recalculate theta
        cz,cx,cy=self.c
        x,y,z=self.Ixi-cx,self.Iyi-cy,self.Izi-cz
        Idist,Itheta,Ipsi=cart2polar(x,y,z)
#        Iangle=(Itheta.flat[idx],Ipsi.flat[idx])
        Iangle_=(Itheta,Ipsi)
        
        parIangle_=((ite,ips) for ite,ips in zip(Itheta.flatten(),Ipsi.flatten()))
        
        pool=ThreadPool(self.n_jobs)        
        
        R_a=np.sum([aml*Y(w,Iangle_) for aml,w in zip(self.Aml,self.Freq)],0)

#        self.it=product( zip(self.Aml,self.Freq ),parIangle_    )
        print 'got here'
#        results=pool.map( dummy ,  product( zip(self.Aml,self.Freq ),parIangle_    )  )
#
#        results=pool.map( parY ,  product( zip(self.Aml,self.Freq ),parIangle_    )  )


#        self.it=product( parIangle_ ,  zip(self.Aml,self.Freq) ) #not this
        self.it=product( zip(self.Aml,self.Freq) ,parIangle_    )  #This is the correct order!

        results=pool.map( dummy , self.it)
#        results=pool.map( dummy , product( parIangle_ ,  parIangle_ ))
       
#        Results=self.results.reshape( self.nF, len(z)*len(x)*len(y) )
        print 'and here'
        
        
        self.R_a=R_a
        self.results=results
#        self.Results=Results
        
        
        
#        Yw=partial(Y,angle=Iangle_)
#        results=pool.map(Yw,[w for w in self.Freq])
#        self.Results=results
##        results=pool.map_async(Yw,[w for w in self.Freq])
#        output=[p for p in results]

#        output=np.array([p.get() for p in results])
#        self.Output=output        
        pool.close()
        pool.join()
        
        R0=self.r0*np.ones(self.I.shape)
    #    R=R0+R_a+R_b
        R=R0+R_a
        self.phi=Idist-R
        print 'total time for iter is ',time.time()-start    

        self.PhiRecord.append(self.phi)


if __name__ == '__main__':
    freeze_support()
     
    
    marker=37
    
    path='/work/03176/csnyder/Volumes/150226 Hi-res_40x W5/patches/patch'+str(marker)+'.tif'
    
    I=skio.imread(path)
    I=I.astype(np.float64)
    I/=I.max()

    spacing=np.array([2.0,1.0,1.0])#300x100x100um #150226 Hi-res_40x


    shape=Shape(I,spacing)
    print 'shape init'
    
#    for i in range(shape.n_iter):
    for i in range(shape.n_iter):
        shape.do_iter()
    








#######Holy crap so pissed this didn't matter to optimize:########
#######Holy crap so pissed this didn't matter to optimize:########
#######Holy crap so pissed this didn't matter to optimize:########


#        A_tup=[('coef',i,freq_ml,Iangle,distributedF,w_ml,a_ml,self.lamL,self.tau) for i,freq_ml,w_ml,a_ml in zip(range(self.nF),self.Freq,self.Wml,self.Aml)]
#        C_tup=[('center',i+self.nF,cwt[:,i],distributedF,spacing[i],self.c[i],self.c_est[i],self.lamM,self.tau) for i in range(len(self.c))]
#        R_tup=[('radius',3+self.nF,distributedF,self.r0,self.r_est,self.lamR,self.tau )]
#
##        pool=mp.pool(self.n_jobs)
#        pool=ThreadPool(self.n_jobs)
#        result=pool.map( calc_delta,A_tup+C_tup+R_tup )
#        result=np.array(result)
#        
#        self.result_order=result[:,0]
#        dF=result[:,1]
#        
#        
#        self.dF=dF
#        
#        self.Aml-=dF[:self.nF]
#        self.c -= dF[self.nF:self.nF+3]
#        self.r0 -= dF[self.nF+3]
#
#def do_coefs(my_tuple):
#    index,freq_ml,Iangle,distributedF,w_ml,a_ml,lamL,tau=my_tuple
#    
#    awt=Y(freq_ml,Iangle)
#    da=np.sum([wt*force for wt,force in zip(awt,distributedF)])
#    F_a=2* w_ml * a_ml
#    da+=lamL*F_a        
#    return (index,tau*da)
#    
#def do_center(my_tuple):
#    index,cwt_i,distributedF,scale,c_i,c_est_i,lamM,tau=my_tuple
##        cwt_i=cwt[:,i]
#    dc_i=np.sum([wt*f for wt,f in zip(cwt_i,distributedF)])
#    F_M=2*(c_i-c_est_i) * scale
#    dc_i+=lamM * F_M
##        self.c[i]-=selftau*dc
#    return (index,tau*dc_i)
#    
#def do_radius(my_tuple):
#    index,distributedF,r0,r_est,lamR,tau=my_tuple
#    delr0=np.sum(distributedF)
#    F_R=2.0 * (r0-r_est)
#    delr0+=lamR *F_R
##        self.r0-=self.tau*delr0
#    return (index,tau*delr0)
#
#
#def calc_delta(big_tuple):
#    data_type=big_tuple[0]
#    my_tuple=big_tuple[1:]
#    
#    if data_type=='coef':
#        return do_coefs(my_tuple)
#    elif data_type=='center':
#        return do_center(my_tuple)
#    elif data_type=='radius':
#        return do_radius(my_tuple)
#    else:
#        return 'why is this spot missing'

#######Holy crap so pissed this didn't matter to optimize:########
#######Holy crap so pissed this didn't matter to optimize:########
#######Holy crap so pissed this didn't matter to optimize:########
##################################################################




#index,freq_ml,Iangle,distributedF,w_ml,a_ml,lamL,tau=my_tuple
#index,cwt_i,distributedF,spacing,c_i,c_est_i,lamM,tau=my_tuple
#index,distributedF,r0,r_est,lamR,tau=my_tuple
    
    


#        pool=Pool
        
#        ####  G  ###
#        #NLP=normalized_laplacian(phi)
#        #I_G=lamG*Ig*NLP#wastefully calculates for entire image first
#        #F_G=I_G.flat[idx]
#           
#        #distributedF = lamP*F_P + lamG*F_G
#        
#        delA=np.array([np.sum([wt*force for wt,force in zip(al_wt,distributedF)]) for al_wt in Awt])
#        delc=np.sum([wt*f for wt,f in zip(cwt,distributedF)],0)
#        delr0=np.sum(distributedF)
#        
#        F_L_a=np.array([2*wl*al for wl,al in zip(self.Wml,self.Aml)])   
#        F_M=2*(self.c-self.c_est) * spacing
#        F_R=2.0 * (self.r0-self.r_est)
#        
#        delA+= self.lamL * F_L_a        #;delB+= lamL * F_L_b
#        delc+=self.lamM * F_M
#        delr0+=self.lamR *F_R
#    
#        self.Aml-=self.tau*delA
#        self.r0-=self.tau*delr0
#        self.c -=self.tau*delc








#########junk###


        
    
    
    
    
    
    
    
    
#            
#        idx=getbdry(phi,delta)
#        u1pts = np.flatnonzero(phi<=0)         # interior points
#        u0pts = np.flatnonzero(phi>0)          # exterior points
#        u1 = np.sum(I.flat[u1pts])/(len(u1pts)+eps) # interior mean
#        u0 = np.sum(I.flat[u0pts])/(len(u0pts)+eps) # exterior mean
#
#    
#        cz,cx,cy=c
#        x,y,z=Ixi-cx,Iyi-cy,Izi-cz
#        Idist,Itheta,Ipsi=cart2polar(x,y,z)
#        Iangle=(Itheta.flat[idx],Ipsi.flat[idx])
#        Iangle_=(Itheta,Ipsi)
#
#        Awt=[Y(w, Iangle ) for w in Freq]
#        cwt=np.hstack([z.flat[idx][:,np.newaxis], x.flat[idx][:,np.newaxis], y.flat[idx][:,np.newaxis]])
#        F_P=np.array([ (I.flat[ix]-u1)**2-(I.flat[ix]-u0)**2   for ix in idx])
#    
#    
#        ####  G  ###
#        #NLP=normalized_laplacian(phi)
#        #I_G=lamG*Ig*NLP#wastefully calculates for entire image first
#        #F_G=I_G.flat[idx]
#           
#        #distributedF = lamP*F_P + lamG*F_G
#        distributedF = lamP*F_P
#        
#        delA=np.array([np.sum([wt*force for wt,force in zip(al_wt,distributedF)]) for al_wt in Awt])
#        delc=np.sum([wt*f for wt,f in zip(cwt,distributedF)],0)
#        delr0=np.sum(distributedF)
#        
#        F_L_a=np.array([2*wl*al for wl,al in zip(Wml,Aml)])   
#        F_M=2*(c-c_est) * spacing
#        F_R=2.0 * (r0-r_est)
#        
#        delA+= lamL * F_L_a        #;delB+= lamL * F_L_b
#        delc+=lamM * F_M
#        delr0+=lamR *F_R
#    
#        Aml-=tau*delA
#        r0-=tau*delr0
#        c -=tau*delc
#    
#        if debug==1:
#            print '\niteration ',iteration    
#            print 'u0 is ',u0,' u1 is ',u1
#            print 'len(idx) is ',len(idx)    
#            print 'delr0 is ',delr0
#            print 'delc is ', delc
            
        #Idist,Itheta=cart2polar(Ixi,Iyi,c)
        
        ##update c,r0 at some point
        ##recalculate theta
#        cz,cx,cy=c
#        x,y,z=Ixi-cx,Iyi-cy,Izi-cz
#        Idist,Itheta,Ipsi=cart2polar(x,y,z)
#        Iangle=(Itheta.flat[idx],Ipsi.flat[idx])
#        Iangle_=(Itheta,Ipsi)
#        
#        
#        
#        R_a=np.sum([aml*Y(w,Iangle_) for aml,w in zip(Aml,Freq)],0)
#    
#        
#        R0=r0*np.ones(I.shape)
#    #    R=R0+R_a+R_b
#        R=R0+R_a
#        phi=Idist-R
        
        
        
#        self.PhiRecord.append(phi)

#########################



#from mayavi import mlab
#from mayavi.api import Engine
#engine = Engine()
#engine.start()
#
#sIzi,sIxi,sIyi=spacing[0]*Izi,spacing[1]*Ixi,spacing[2]*Iyi
#
#def InspectScene(n):
#    phi=PhiRecord[n]
#    engine.new_scene()
#    mlab.pipeline.volume(mlab.pipeline.scalar_field(sIzi,sIxi,sIyi,I))
#    volume=engine.scenes[-1].children[0].children[0].children[0]
##    volume.volume_mapper.blend_mode = 'maximum_intensity'
#    mlab.contour3d(sIzi,sIxi,sIyi,phi,color=(1,0,0),opacity=0.25,contours=[0])    
#    mlab.points3d(spacing[0]*c_est[0],spacing[1]*c_est[1],spacing[2]*c_est[2])
#
#
#InspectScene(n_iter)
#mlab.text(.05,.02,'150302_CM_Hi-res_40x W5',width=0.3)
#mlab.text(.05,.07,'marker '+str(marker),width=0.15)
#mlab.text(.05,.10,'iter '+str(n_iter),width=0.08)
#dst_dir='/home1/03176/csnyder/temp_output'
#nsaves=0
#
#
#
#mlab.savefig(dst_dir+'/150302_Hi-res_40x W5 m' +str(marker)+' 0'+str(nsaves)+'.png');nsaves+=1




mlab.savefig(dst_dir+'/m' + str(marker)+' Very congested 20Xave attempt2- lamM=35 04' +'.png')


#if __name__ == '__main__':

#    fundata=range(10)
#           
#    from multiprocessing import Pool as ThreadPool
#    pool=ThreadPool(n_jobs)
#    multiproc_results=pool.map(fun,fundata)
#    pool.close()
#    pool.join()
#           
#    
#    out=[fun(datum) for datum in fundata]
    
#    from joblib import Parallel, delayed
#    joblib_results=Parallel(n_jobs=n_jobs,verbose=1)(delayed(fun)(datum) for datum in fundata)

    