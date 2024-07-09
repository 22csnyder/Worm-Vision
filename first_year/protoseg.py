# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 07:33:40 2015

@author: christopher

This is a prototype of alignment and identification through
reproducibility of segmentation

A set of images of possibly dimensions, (but same scale!)
are fed into the algorithm. The algorithm proceeds by segmentation
and tries to align parts of the images by generating 'landmark'
phrases where possible.

"""
from skimage import img_as_float
import skimage.io as io
from ioworm import ls,return_those_ending_in
import numpy as np
#import cv2
from skimage.segmentation import felzenszwalb
from skimage.segmentation import mark_boundaries
from functools import partial
from mlworm import getSegmentFeatures,computeKernel,gaussianKernel,relativeGaussianKernel
import math
import matplotlib.pyplot as plt
from WormPlot import sidebysideplot
from WormFuns import lmap,lzip,returnGrid

#Helper functions
def closestFactors(n):
    fact=[[i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0]
    c=n
    d=[f for f in fact if abs(f[0]-f[1])<=c]
    return d[-1]
    
def pairs(alist):
    alist=list(alist)#incase np.array
    dst=[]
    for i,a in enumerate(alist):
        newlist=alist[(i+1):]
        for b in newlist:
            dst.append((a,b))
    return dst

def Kplot(setofK,u=-1):
    if len(setofK)==1:
        if u==-1:
            plt.figure()
        plt.imshow(setofK[0],interpolation='none')
        return
    [n,m]=closestFactors(len(setofK))
    fig,axes=plt.subplots(m,n)
#    if m==n==1:
#        axes=axes[:,0]
#    else:
    axes=axes.flatten()
#    titles=['0','1','2','3','4','5']
    fig.subplots_adjust(hspace=.001,wspace=0.001)
    fig.tight_layout(pad=0,w_pad=0.0001,h_pad=0.0001)

#    for p,a,t in zip(setofK,axes,titles):
    for p,a in zip(setofK,axes):
        a.imshow(p,vmin=p.min(),vmax=p.max(),interpolation='none')
#        a.set_title(t)
    fig.canvas.draw()

#This function tries to revise the similarity between two elements based on higher order interactions

def scored(K,d1,d2):
    m,n=K.shape
    _K=K.copy()
    def f(xx,d):
        x=np.expand_dims(xx,1)
        loc=np.array(d)
        mew=np.sum(loc*x,0)/np.sum(x)#broadcasting
        var=np.sum(((loc-mew)**2)*x)/np.sum(x)
        return math.sqrt(var)
    for i in range(m):
        _K[i,:]/=f(K[i,:],d2)
    for j in range(n):
        _K[:,j]/=f(K[:,j],d1)
        
    means=[[np.mean(k) for k in _K],[np.mean(k) for k in _K.transpose()]]
    for i in range(m):
        for j in range(n):
            _K[i,j]/=math.sqrt(means[0][i]*means[1][j])
            
#    _K/=_K.mean()
    return _K
#
def iterateScoring(pairK,Loc):
    pairwise=pairs(range(len(Loc)))
    spairK=[scored(K,Loc[p[0]],Loc[p[1]]) for K,p in zip(pairK,pairwise)]
    return spairK    




####Main class####

class Registrar:
    def __init__(self,fn,slow=-1):
        if isinstance(fn,str):
            self.v=self.ReadImages(fn)#get all .tif in directory
        elif isinstance(fn,list):
            if isinstance(fn[0],np.ndarray):
                self.v=fn
        self.nW=len(self.v)
        self.dim=len(self.v[0].shape)
        self.H=1 if self.v[0].shape[0]<self.v[0].shape[1] else 0
        self.pairwise=pairs(range(self.nW))
        self.score_number=0#number of scoring iterations on K
        if slow==-1:                
            self.do_segment()
            self.do_feature()
            sig=50        
            self.setKernel(sig)
            self.computeKernel()
#            self.score()#optional
            self.computeDistances()
            self.findBestAlignment()
    def ReadImages(self,foldername):
        a=ls(foldername)
        data_tif=return_those_ending_in(a,'tif')
        all_imgs=[img_as_float(io.imread(im,'as_grey')) for im in data_tif]
        v=[im/im.max() for im in all_imgs]#scale to 0-1 32bit float        
        print("Images Read")
        return v
    def do_segment(self,scale=-1):
#            self.seg=partial(felzenszwalb,sigma=0,min_size=5)#OPTION
            self.seg=partial(felzenszwalb,sigma=3,min_size=50)
            if scale==-1:
                scale=1000
#            sca=10
            self.segments=[self.seg(img,scale=scale) for img in self.v]
            print('Segmentation Complete')
    def do_feature(self):
            self.abs_split=self.dim
            self.Fea=[getSegmentFeatures(im,s) for im,s in zip(self.v,self.segments)]
            self.sort_features()
            self.Loc=[[clusterfeature[:self.abs_split] for clusterfeature in worm] for worm in self.Fea]
            print("Features Computed and Sorted")
    def show_segmentation(self):
        v_bdd=[mark_boundaries(im,s) for im,s in zip(self.v,self.segments)]
        titles=[str(a) for a in range(self.nW)]
        return sidebysideplot(v_bdd,titles,horiz=self.H)
    def sort_features(self):
        def xloc(f):
            return f[self.H]#clever sort
        for f in self.Fea: f.sort(key=xloc)
    def setKernel(self,sig):#put sig at -1 to neglect distances all together  
#        A_sig=75#for distance
        A_sig=sig
        kerA=partial(gaussianKernel,sigma=A_sig*np.ones(self.abs_split))
        kerR=partial(relativeGaussianKernel,sigma=np.ones(self.Fea[0][0].__len__()-self.abs_split))
        def ker(x,y,A_sig):
            x1=x[:self.abs_split];x2=x[self.abs_split:]
            y1=y[:self.abs_split];y2=y[self.abs_split:]
            
            return kerR(x2,y2) if A_sig==-1 else kerA(x1,y1)*kerR(x2,y2)
        self.ker=partial(ker,A_sig=A_sig)
    def computeKernel(self):
        self.Klist=[computeKernel(self.Fea[p[0]],self.Fea[p[1]],self.ker) for p in self.pairwise]
        self.origKlist=[np.copy(K) for K in self.Klist]
        print("Kernel Computed")
    def score(self):
        self.Klist=iterateScoring(self.Klist,self.Loc)
        self.score_number+=1
        print("scoring comlete")
    def computeDistances(self):
        self.xyzDlist=[]#each entry keeps track of a certian dimensions offsets
        for d in range(self.dim):
            Dlist=[]
            for p in self.pairwise:
                L1=self.Loc[p[0]];L2=self.Loc[p[1]]
                l1=list(zip(*L1))[d];l2=list(zip(*L2))[d]#pick which dimension to work on
                m,n=len(l1),len(l2)
                D=np.zeros((m,n))
                for i in range(m):
                    for j in range(n):
                        D[i,j]=l1[i]-l2[j]
                Dlist.append(D)
            self.xyzDlist.append(Dlist)
    def findBestAlignment(self):
        Alist=[np.sum(K) for K in self.Klist]
        def bdds(p,l):
            if p[0]<l<=p[1]: return True
            else: return False
        r=[]
        for Dlist in self.xyzDlist:
            Blist=[np.sum(K*D) for K,D in zip(self.Klist,Dlist)]#K*D is entrywise mult
            #Ux=b
            b=[sum([B for B,p in zip(Blist,self.pairwise) if bdds(p,l)==True]) for l in range(1,self.nW)]
            U=[]
            for l in range(1,self.nW):
                a=[sum([A for A,p in zip(Alist,self.pairwise) if bdds(p,l)==True and bdds(p,k)==True]) for k in range(1,self.nW)]
                U.append(a)
            
            U=np.array(U);b=np.array(b)
            r.append(np.linalg.lstsq(U,b)[0])
        self.R=r
        print("current alignment after ",self.score_number," scoring iterations:")
        for r in self.R: print(r)
        return self.R
    def score_iteration(self):
            self.score()#optional
            self.computeDistances()
            self.findBestAlignment()
            
#        self.R=
#        return r    

    






A=np.zeros((20,40))
B=A.copy()
import skimage.draw as skdraw
rr3,cc3=skdraw.circle(5,10,3)
rr5,cc5=skdraw.circle(13,18,5)
A[rr3,cc3]=1
A[rr5,cc5]=1
B[rr3,cc3+12]=1
B[rr5,cc5+12]=1


path='./../WormPics/Synthesized/MegaplusTestData'
#path='./../WormPics/Synthesized/2DtestData'
#regis=Registrar([A,B],1)#OPTION
regis=Registrar(path,1)#The 1 let's us do things manually


regis.do_segment(scale=10000)
#fig,axes=regis.show_segmentation()
regis.do_feature()
#sig=50#mutable     
dist_sig=-1#neglect distance contribution   
regis.setKernel(dist_sig)
regis.computeKernel()
#regis.score()#optional
regis.computeDistances()
regis.findBestAlignment()


Kplot(regis.origKlist)
plt.title('orig')

sidebysideplot(regis.segments,colormap=-1)
regis.show_segmentation()
#regis.score_iteration();Kplot(regis.Klist);plt.title(str(regis.score_number))



#Vote based registration:
#Define grid:

print("Voting Grid Method")

class GridRules:
    def __init__(self,size,spacing):
        self.spacing=spacing
        self.size=size
        Grid1=returnGrid(2*size,self.spacing)
        self.Grid=np.array([g-np.mean(g) for g in Grid1])
        self.bin_centers=np.array([[(G[i]+G[i+1])/2 for i in range(len(G)-1)] for G in self.Grid])
        #For 2 dimensions..will have to change for 3
        self.Hist=np.zeros((len(self.Grid[0])-1,len(self.Grid[1])-1))#H[0] is the space between G[0] and G[1]
    def binIt(self,Sim,Dxyz,ilessj):
        row,col=Sim.shape        
        n=row if ilessj else col
        tempHist=[]
        for D,G in zip(Dxyz,self.Grid):
            H=np.zeros(len(G)-1)
            for i in range(n):
                dist=D[i,:] if ilessj else D[:,i]
                seg=Sim[i,:] if ilessj else Sim[:,i]
                q=np.histogram(dist,bins=G,weights=seg)[0]#segmentcontributes weights into each bin.
                #q/=np.sum(q)#optional                        
                H+=q
            tempHist.append(H)
        for H in tempHist:
            H/=H.sum()
        return tempHist

    def pairWiseP(self,Sim,Dxyz):
        Hi=self.binIt(Sim,Dxyz,ilessj=1) 
        Hj=self.binIt(Sim,Dxyz,ilessj=0)        
        self.Histogram=[hi+hj for hi,hj in zip(Hi,Hj)]
#        return self.Histogram        


class PairwiseGrid(GridRules):
    def __init__(self,Sim,Dxyz,size,grid_spacing):
        super().__init__(size,grid_spacing)
        self.P=self.pairWiseP(Sim,Dxyz)
    

size=2*np.array(regis.v[0].shape)
grid_spacing=(0.05*size).astype('int')
PWgrid=partial(PairwiseGrid,size=size,grid_spacing=grid_spacing)

#Sim=regis.Klist[0]
#Dxyz=[d[0] for d in regis.xyzDlist]#first pair, ever dimension
#D=Dlist[0]


#Define voting histograms for each pair
DxyzList=[[d[ix] for d in regis.xyzDlist] for ix in range(len(regis.pairwise))]
GridList=[PWgrid(Sim,Dxyz) for Sim,Dxyz in zip(regis.Klist,DxyzList)]


#(for a visual)Pick the best offset for each pair
print("early pairwise prediction")
Align=[]
Alignxyz=[]
for Gridxyz in GridList:#for each pair of worms
    Alignxyz.append([G[H==H.max()][0] for G,H in zip(Gridxyz.bin_centers,Gridxyz.Histogram)])#Do each dimension

for a,p in zip(Alignxyz,regis.pairwise):
    print(a,p)
    

#Now try big picture
Histlist=[Gridxyz.Histogram for Gridxyz in GridList]

#multiH=

#for Ii in range(regis.nW):
#    comparelist=[[i,[pi for pi in p if not pi==Ii][0]] for i,p in enumerate(regis.pairwise) if (Ii in list(p))]
#    Similiarity=[]
#    for s in range(len(regis.Fea[Ii])):#Going through segments in image Ii
#        for pos,Ij in comparelist:#Go through other images
#            K=np.copy(regis.Klist[pos])
#            Dxyz=[d[pos] for d in regis.xyzDlist]
#            #Opportunity to normalize suitably before voting
##Ideas:k most informative segments
##Normalize each segment contribution?            
#            if Ii<Ij:
#                Similiarity.append(K[s,:])
#            else:
#                Similiarity.append(K[:,s])
            



