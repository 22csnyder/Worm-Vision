# -*- coding: utf-8 -*-
"""
Created on Thu May 21 06:58:20 2015

@author: Christopher Snyder
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


def setup(**Params):
    alpha=Params.get('alpha',None)
    beta=Params.get('beta',None)
    npoints=Params.get('npoints',None)
    init_radius=Params.get('init_radius',None)
#    print Params
    midpoint=np.array([35,35])
#    npoints=16###DEBUG###
    points=[]
    for i in xrange(npoints):
        x=midpoint[0]+init_radius*math.cos(2*math.pi*i/npoints)
        y=midpoint[1]+init_radius*math.sin(2*math.pi*i/npoints)
        points.append(np.array([int(x),int(y)]))    
    points=np.array(points)    
    a,b=alpha,beta
    base=[-1*b,a+4*b,-2*a-6*b,a+4*b,-1*b]
    base.extend([0 for i in range(npoints-5)])
    base=np.array(base)
    rebase=np.roll(base,-2)
    A=circulant(rebase)    
    M=np.eye(npoints)-h*A
    return points,M
    
#def setup(**Params):
#    alpha=Params.get('alpha',None)
#    beta=Params.get('beta',None)
#    npoints=Params.get('npoints',None)
#    print npoints
    
    
    
#img=Slices[neuron_n].astype(np.float)
#Preprocess to [0,1]
#img/=img.max()
#img=autoFijiStyleScaling(img)


#    npoints=Params.get('npoints',None)
def placeLines(drawimg,points):#AFAIK **Params not needed
    npoints=len(points)
    pool=cycle(points)
    nlines=0
    p=pool.next()
    q=pool.next()
    wt=drawimg.max()*1.5
    while nlines<npoints:
        ####PLEASE note that opencv uses x,y not row,col
        cv2.line(drawimg,(p[1],p[0]),(q[1],q[0]),(wt,0,0))
        p,q=q,pool.next()
        nlines+=1
    return drawimg
    
    
#drawimg=img.copy()
##See if initial lines are okay:
#placeLines(drawimg,points)
#squarebysideplot([img,drawimg])

def calculateForces(img,**Params):
    edge_scale=Params.get('edge_scale',None)
    NRG_method=Params.get('NRG',None)
    
    gau_img=gaussian_filter(img,3)
    
    if NRG_method==0:
    #    Sx=-1.0*sobel_h(gau_img)#Pics out horizonal edges
    #    Sy=-1.0*sobel_v(gau_img)#Pics out vertical edges (along a column)
        Fx=-1.0*sobel_h(gau_img)
        Fy=-1.0*sobel_v(gau_img)    
        Fx*=edge_scale
        Fy*=edge_scale
        return Fx,Fy    
    
    elif NRG_method==1:
        E=-1.0*sobel(gau_img)**2
    #    Sx=-1.0*sobel_h(gau_img)#Pics out horizonal edges
    #    Sy=-1.0*sobel_v(gau_img)#Pics out vertical edges (along a column)
        #    Fx=-1.0*sobel_h(gau_img)
        #    Fy=-1.0*sobel_v(gau_img)
        Fx=-1.0*sobel_h(E)
        Fy=-1.0*sobel_v(E)
        
        Fx*=edge_scale
        Fy*=edge_scale
        return Fx,Fy

####Section for exploring ####
#m=103
#m_idx=44

#m=225
#
#img=Slices[m].astype(np.double)
##img/=img.max()
#gau_img=gaussian_filter(img,3)
#E=-1.0*sobel(gau_img)**2
#Sx=-1.0*sobel_h(gau_img)#Pics out horizonal edges
#Sy=-1.0*sobel_v(gau_img)#Pics out vertical edges (along a column)
##    Fx=-1.0*sobel_h(gau_img)
##    Fy=-1.0*sobel_v(gau_img)
#Fx=-1.0*sobel_h(E)
#Fy=-1.0*sobel_v(E)
#Equil=placeLines(E.copy(),PointsDraw[850])
##Equil=placeLines(E.copy(),Result[m_idx])
#squarebysideplot([img,gau_img,Sx,Sy,E,Fx,Fy,Equil],['img103','gau_img','Sx','Sy','E','Fx','Fy','Equil'],['Greys'])


#arr_neuron_n=np.array(neuron_n)
#np.where(arr_neuron_n==m)

#Fx=1.0/(Sx+1)
#Fy=1.0/(Sy+1)
#    edge_sobel = sobel(img)
#    gedge_sobel = sobel(gau_img)
#MagGrad=( (Fx**2)+(Fy**2))**0.5

#med=np.median(img)

#lvl1=839
#lvl2=3799
#gau1=gaussian_filter(Fy,1)
#gau3=gaussian_filter(Fy,3)
#gau5=gaussian_filter(Fy,6)
#dst=np.zeros(img.shape)
#dst[img<=lvl1]=gau1[img<=lvl1]
#dst[(lvl1<img)*(img<=lvl2)]=gau3[(lvl1<img)*(img<=lvl2)]
#dst[img>lvl2]=gau5[img>lvl2]

#Dummmy tests:
#a=np.linspace(-5,5,75)
#a=np.dot(np.ones((75,1)),a.reshape(1,75))
#Fx=a.transpose()
#Fy=a
#Fx=np.zeros((75,75))
#Fy=np.zeros((75,75))

#look at forces:
#squarebysideplot([Sx,Sy,Fx,Fy],['Sx','Sy','Fx','Fy'],['Greys'])

#Convolve
#Fx=gaussian_filter(Fx,10)
#Fy=gaussian_filter(Fy,10)



def getSurfaceNormals(points):
    left=np.roll(points,1,0)
    right=np.roll(points,-1,0)
    return points-0.5*(left+right)

from skimage.draw import polygon
def getInteriorIntensity(X):
    x=X[:,0]
    y=X[:,1]
    return np.mean(img[polygon(x.astype(np.int),y.astype(np.int))])

def getProbability(points,nbins=10):
    points=np.array(points)
    x=points[:,0]
    y=points[:,1]
    data=img[polygon(x,y)]
    hi,bi=np.histogram(data,bins=nbins)
    hi=hi.astype(np.float)
    hi/=len(data)
    def p(I):
        eps=0.01
        if I<bi[0]:
            return eps
        if I>bi[-1]:
            return eps
        idx=np.where( (bi[:-1]>=I) )
        if hasattr(idx,"__iter__"):
            idx=idx[0]
        return np.sum(hi[idx])
    return p


#from sklearn.cluster import KMeans
#est=KMeans(n_clusters=2)
#est.fit(img.reshape(-1,1))
#labels=est.labels_.reshape(75,75)
#H0,B0=np.histogram(img[labels==0])
#H1,B1=np.histogram(img[labels==1])
#H0,H1=H0.astype(np.float),H1.astype(np.float)
#H0/=np.sum(H0)
#H1/=np.sum(H1)
#
#if np.sum(B0)>np.sum(B1):#Make sure 0 label is background, 1 label is foreground
#    H0,H1=H1,H0
#    B0,B1=B1,B0

def value(Intensity):
#    sp0=B0[1]-B0[0]
#    sp1=B1[1]-B1[0]
    if Intensity<=B0[-1] and Intensity>=B0[0]:
        idx0=np.where( (B0[1:]>=Intensity)*(B0[:-1]<=Intensity) )
        if hasattr(idx0,"__iter__"):
            idx0=idx0[0]
        if idx0==len(H0):
            prob0=1
        else:
            prob0=np.sum(H0[idx0])
    else:
        prob0=0
    if Intensity<=B1[-1] and Intensity>=B1[0]:
        idx1=np.where( (B1[:-1]<=Intensity)*(B1[1:]>=Intensity) )
        if hasattr(idx1,"__iter__"):
            idx1=idx1[0]
        if idx1==len(H1):
            prob1=1
        else:
            prob1=np.sum(H1[idx1])
    else:
        prob1=0
    return prob1/(prob1+prob0) - 0.5
    
    

#def value(Intensity):
#    spacing=ImgBins[1]-ImgBins[0]
#    idx=np.where( (bi>I)*(bi<=I+spacing) )
##    histsums=
#    return np.percentile(img.ravel(),Intensity)-0.5



####Attribute a "value" to each pixel####
#ImgHist,ImgBins=np.histogram(img.ravel(),20)
#total=np.sum(ImgHist)
#HistSums=[np.sum(ImgHist[:n+1]) for n in range(len(ImgHist))]
#def value(Intensity):
#    spacing=ImgBins[1]-ImgBins[0]
#    idx=np.where( (bi>I)*(bi<=I+spacing) )
##    histsums=
#    return np.percentile(img.ravel(),Intensity)-0.5
    




def F3(Xtm1):
    P=getProbability(Xtm1)        
    force_x,force_y=[],[]
    J=getInteriorIntensity(np.array(Xtm1))
    V=getSurfaceNormals(Xtm1)
    for p,v in zip(Xtm1,V):
        Inten=img[int(p[0]),int(p[1])]
        val=value(Inten)
        sca=value(Inten)*P(Inten)
        aa=0.1
        mag=math.sqrt(np.sum(v**2))
        v/=mag
        dif=(img[p[0],p[1]]-J)        
        sca1=dif/(1 + aa* (dif**2))
        sca+=sca1
        force_x.append(sca*v[0])
        force_y.append(sca*v[1])
    return np.array(force_x),np.array(force_y)
    
        
def F2(Xtm1):
    force_x,force_y=[],[]
    J=getInteriorIntensity(np.array(Xtm1))
    V=getSurfaceNormals(Xtm1)
    for p,v in zip(Xtm1,V):
        aa=0.1
        mag=math.sqrt(np.sum(v**2))
        v/=mag
        dif=(img[p[0],p[1]]-J)        
        sca=dif/(1 + aa* (dif**2))
        force_x.append(sca*v[0])
        force_y.append(sca*v[1])
        
#        sca=dif/(MagGrad[p[0],p[1]]**2 + 0.1* (dif**2))
#        force_x.append(sca*Fx[p[0],p[1]])
#        force_y.append(sca*Fy[p[0],p[1]])
    return np.array(force_x),np.array(force_y)    


def F(Xtm1,Fx,Fy):
    force_x,force_y=[],[]
    for p in Xtm1:
        force_x.append(Fx[p[0],p[1]])
        force_y.append(Fy[p[0],p[1]])
    return np.array(force_x),np.array(force_y)



def doIter(xold,yold,img,M,**kwargs):
    if kwargs is not None:
        Fx=kwargs.get('Fx',None)
        Fy=kwargs.get('Fy',None)
    Force_x,Force_y=F(zip(xold,yold),Fx,Fy)
    xnew=solve(M,xold+h*Force_x)
    ynew=solve(M,yold+h*Force_y)
    dimx,dimy=img.shape
    xnew[xnew>(dimx-1)]=(dimx-1)
    ynew[ynew>(dimy-1)]=(dimy-1)
    xnew[xnew<0]=0
    ynew[ynew<0]=0
    return xnew,ynew

def computePoints(img,iteration_mode=False,**Params):
    n_iter=Params.get('n_iter',None)
    points,M=setup(**Params)
    Fx,Fy=calculateForces(img,**Params)
    x,y=zip(*points)
    x,y=np.array(x),np.array(y)
    PointsObs=[];PointsDraw=[]
    PointsObs.append(points)#original
    PointsDraw.append(zip(x.astype(np.int),y.astype(np.int)))
    iteration=0
    while iteration<n_iter:
        iteration+=1
        x,y=doIter(x,y,img,M,Fx=Fx,Fy=Fy)
        if iteration_mode:
            PointsObs.append(zip(x,y))
            PointsDraw.append(zip(x.astype(np.int),y.astype(np.int)))    
    if iteration_mode:
        return PointsObs,PointsDraw
    else:
        return zip(x,y) , zip(x.astype(np.int),y.astype(np.int))


def do_main_loop(data_in):#only called when n_jobs>1
    n,img,Params=data_in
    print 'starting on neuron ',n
    _,result=computePoints(img)    
    drawimg=placeLines(img.copy(),result)
    Draw_List.append( (n,drawimg) )#to reorder later
    Result.append( (n,result) )


if __name__ == '__main__':#Necessary for joblib.Parallel
#############################################################################
#####################   Settings and Load Data    ###########################
    pathWormPics='./../Corral/Snyder/WormPics'
    WormConfig = {
        'WormPics directory':pathWormPics,
        'working directory':pathWormPics+'/Synthesized/truncatedStitch/150226 Hi-res_40x/W5',
        'data set identifier':'150226 Hi-res_40x W5'
    }
    Id=WormConfig['data set identifier']
    #Currently loads patch19
    #patch=np.load(WormConfig['working directory']+'/'+WormConfig['data set identifier']+'.npy')
    #img=patch[neuron_n]
    #Or just load slices for now:
    #my_marker_no=[68,72,80,81,85,89,102,103,112,118,120,143,163]
    #mySlices=[Slices[n] for n in my_marker_no]
    Slices=np.load(WormConfig['working directory']+'/'+'Slices.npy')
###########################################################################
###################           Parameters          #########################

    ##Program Mode parameters to be determined later    
    UNDEFINED=object()
    
    iteration_mode=UNDEFINED
    plot_in_pieces=UNDEFINED
#    plot_output_plots=UNDEFINED
    iteration_mode=UNDEFINED
    iter_numbers=UNDEFINED        
    #********************#######EDIT BELOW HERE#######*********************#
    chunk_size=59#For plotting
    
    Params={
    #If neuron_n has len>59, several plots are made
#    'neuron_n':range(0,230)
#    'neuron_n':range(59,117)
#    'neuron_n':68
#    'neuron_n':72
#    'neuron_n':103
#    'neuron_n':50
    'neuron_n':225
#    'neuron_n':[18,32,34,49,52,103,112,139,205,225]
    #    'neuron_n':range(150,209)
    
    ,'npoints':16 #must be at least 5 for circle
    ,'init_radius':15
    ,'alpha':0.001
    ,'beta':0.001
    ,'h':.1
    ,'edge_scale':10.0
    ,'n_iter':850
    #The NRG parameter describes how the image external forces are administered
    #Originally I mistakenly used essentially the intensity of the image as the 
    #energy, but it sort of worked out well. That's NRG = 0
    #Now I'm trying the negative magnitude of the gradient, NRG = 1
    ,'NRG':1
    }
    #for iter mode    
    iter_numbers=[0,1,5,10,50,200,300,500,700,950,1250,1500,1800,2200]    
    #********************#######EDIT ABOVE HERE#######*********************#
        
    
    #allow for single int entry for neuron_n
    if not hasattr(Params['neuron_n'],'__iter__'):
        Params['neuron_n']=[Params['neuron_n']]

    h=Params['h']
    init_radius=Params['init_radius']
    neuron_n=Params['neuron_n']

    default_iter_numbers=[0,1,3,5,10,50,150,290,400,850]
    if iteration_mode==UNDEFINED:
        if len(neuron_n)==1:
            if iter_numbers==UNDEFINED:
                iteration_mode=default_iter_numbers
            else:
                iteration_mode=iter_numbers
            Params['n_iter']=max(iteration_mode)    
        else:
            iteration_mode=False
    
    #-----------------
    n_jobs=5 #>1 doesn't work right now..
    #-----------------
###########################################################################
###################            MAIN LOOP          #########################
    Result=[]
    Draw_List=[]
    nslices=[Slices[n].astype(np.float) for n in neuron_n]
    X=[im/im.max() for im in nslices]
    

    t0=time.time()#wall time (clock() for process time)  # best precision is seconds :(  
    
    if n_jobs==1:#do loop    
        for n,img in zip(neuron_n,X):
            if iteration_mode:
                print 'returning iterations for neuron ',n
                _,PointsDraw=computePoints(img,iteration_mode=iteration_mode,**Params)
                for ix in iteration_mode:
                    drawimg=placeLines(img.copy(),PointsDraw[ix])
                    Draw_List.append(drawimg)
            else:
                print 'starting on neuron ',n
                _,result=computePoints(img,**Params)
                drawimg=placeLines(img.copy(),result)
                Draw_List.append(drawimg)
                Result.append(result)
    
    elif n_jobs>1:
        from multiprocessing.dummy import Pool as ThreadPool
#        r=Parallel(n_jobs=n_jobs,verbose=1)(delayed(sleep)( 0.1 ) for _ in range(10))
#        r=Parallel(n_jobs=n_jobs,verbose=1)(delayed(do_main_loop)( n,img ) for n,img in zip(neuron_n,X))
        pool=ThreadPool(n_jobs)
        data_in=[(n,img,Params) for n,img in zip(neuron_n,X)]
        main_results=pool.map(do_main_loop,data_in)
        pool.close()
        pool.join()
        
    
    
    total_time=time.time()-t0
    print 'total time is ',total_time,'sec for ',len(neuron_n),' neurons with n_jobs=',n_jobs
###########################################################################
###################            Plotting          ##########################
    
    #Handle plot parameters:
    #(iteration_mode handeled above)
    
    if plot_in_pieces==UNDEFINED:
        if len(neuron_n)>chunk_size:
            plot_in_pieces=True
#            plot_output_plots=True
        else:
            plot_in_pieces=False

#    if plot_output_plots==UNDEFINED:
#        print 'warning plot_output_plots is UNDEFINED'

    #Param Table parameters
    colLabels=('Param','Value')

    if not iteration_mode:
        cellText=[[ke,valu] for ke,valu in Params.iteritems() if ke !='neuron_n']
    else:        
        cellText=[[ke,valu] for ke,valu in Params.iteritems() if ke !='n_iter']
    cellText=[['Id',Id]]+cellText

    #Handle titles
    if iteration_mode:#Titles for iteration
        titles=map(str,iteration_mode)
        titles[0]='iter\nNo. ' + titles[0]
    else:                           #Titles for multiple neurons
        titles=map(str,neuron_n)
        titles[0]+'marker\nNo. '+titles[0]


    
    
    if plot_in_pieces==True:#Split up data
        data_size=len(Draw_List)
        marks_list=range(len(Draw_List))
        
        sublists=[]
        while len(marks_list)>=chunk_size:
            sublists.append(marks_list[:chunk_size])
            marks_list=marks_list[chunk_size:]
        sublists.append(marks_list)
        
        FG=[];AX=[]
        for idx in sublists:
            subtitles=[titles[i] for i in idx]
            subtitles[0]='marker\nNo. '+subtitles[0]
            dr_list=[Draw_List[i] for i in idx]
            fig,axes=squarebysideplot(dr_list)
            for axe,titl in zip(axes,subtitles):
                axe.text(0.1,0.1,titl,verticalalignment='top')
            _table=addTable(fig,ax=axes[-1],cellText=cellText,colLabels=colLabels,loc='center',cellLoc='center')
            FG.append(fig)
            AX.append(axes)
                                   
    else:
        fig,axes=squarebysideplot(Draw_List)
#        fig,axes=sidebysideplot(Draw_List+[np.zeros(Draw_List[0].shape)])
        for axe,titl in zip(axes,titles):
            axe.text(0.1,0.1,titl,verticalalignment='top')
        _table=addTable(fig,ax=axes[-1],cellText=cellText,colLabels=colLabels,loc='center',cellLoc='center')
        
    ##################    ##################    ##################
        
'''        
#Eyeball test bad:
my_marker_no=[68,72,80,81,85,89,102,103,112,118,120,143,163]    

#Snake test bad:
'''
#bad with NRG=0
bad0=[18,32,34,49,52,103,112,139,205,225]

#bad with NRG=1
bad1=[68,72,103]

super_dim=[80,85]

how_am_i_supposed_to_segment_that=[120,139,205]
#'''  
    ##################    ##################    ##################
    
def NTT(num_list):#(Numbers To Title)
    S=list(map(str,num_list))
    long_string=''
    while len(S)>0:
        long_string+=S.pop(0)
        long_string+=','
    return long_string[:-1]

def save_bad_marker(np_img_list,marker_no_list):
    file_name=NTT(marker_no_list)
    dst=np.hstack([np_img_list[i] for i in marker_no_list])
    imsave(WormConfig['working directory']+'/'+file_name+'.tif',dst,compress=1)
    

#ParameterList=['Id','neuron_n','npoints','init_radius','alpha','h','edge_scale','n_iter']
#    neuron_n=range(1,25)
#    #neuron_n=[1,5,10]
#    npoints=16 #must be at least 5 for circle
#    init_radius=15
#    alpha=.10
#    beta=.50
#    h=.1
#    edge_scale=10.0
#    n_iter=850
#    cellText=[[s,eval(s)] for s in ParameterList]
#if isinstance(neuron_n,list):
#    del ParameterList[1]

    
    #plotlist=[placeLines(img.copy(),PointsDraw[n]) for n in numbers]
    #titles=['iter '+str(n) for n in numbers]
    #titles[0]='Neuron '+str(neuron_n)+'...\n'+titles[0]
    
    
    
    #_table.scale(1,1.5)
    #fig.canvas.draw()
    
    #Keys,Cells=zip(*_table.get_celld().items())
    #cell=Cells[0]
    
    #from skimage import draw
    #draw.polygon(np.array([0,0,1,1]),np.array([0,1,0,1]))