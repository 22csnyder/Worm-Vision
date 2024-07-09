# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 13:10:40 2015

@author: christopher
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
try:
    from skimage.filters import threshold_otsu,gaussian_filter
except: #is python3
    from skimage.filter import threshold_otsu,gaussian_filter
import skimage.measure as m #for connected components
from tifffile import imsave


pathWormPics='./../Corral/Snyder/WormPics'
WormConfig = {
    'WormPics directory':pathWormPics,
    'working directory':pathWormPics+'/Synthesized/truncatedStitch/150226 Hi-res_40x/W5',
    'data set identifier':'150226 Hi-res_40x W5'
}
#Read marker script
#ll=ls('./../WormData')
#file_name=ll[1]
#file_name='./../WormPics\\141226_CM_Pan_GCamp_Vol1_Stacks'
#file_name='./../WormData\\141226_ZIM294 HiRes_L4_O20X_F630x75um_P11008x500S173 stack __ Markers150211_attempt2'

ll=ls(WormConfig['WormPics_directory']+'/150226_CM_Hi-res_40x/W5/stacks',1,0)
apofiles=return_those_ending_in(ll,'apo')
file_name=apofiles[0]
f=open(file_name,'r')

file_extension=None
if file_name.split('.')[-1]=='apo':
    file_extension='.apo'
elif file_name.split('.'[-1])=='.marker':
    file_extension='.marker'

#for line in f:
#    print(line.split(',',-1))
def parse_apo_line(line):
    pieces=line.split(',')
    z=int(pieces[4].strip())
    x=int(pieces[5].strip())
    y=int(pieces[6].strip())
    return [x,y,z]
def parse_marker_line(line):
    if line[0]=='#':pass
    else: return list(map(int,line.split(',',-1)[0:3]))
def parse_line(line):
    if file_extension=='.apo':
        return parse_apo_line(line)
    elif file_extension=='.marker':
        return np.array(parse_marker_line(line))
    else:
        raise Exception('file_extension not .apo or .marker')

#li=[line for line in f if line[0]!='#']
marker_list=[parse_line(line) for line in f if line[0]!='#']#in x, y, z#row,col,z





stack_path='./../Corral/Snyder/WormPics/150226_CM_Hi-res_40x/W5/stacks\\150226 Hi-res_40x W5.tif'
#stack_path=ll[1]
#stack_path='./../WormPics\\20150109 for Chris S worm images 2\\141226_ZIM294 HiRes_L4_O20X_F630x75um_P11008x500S173 stack.tif'


def fiji_pims_reorient(frame):#Lets you pretend that x,y in fiji is row,column in spyder
    return np.rot90(frame,3)
def reorient(frame):
    return frame[::-1,...]   
#v=pims.TiffStack(stack_path,process_func=reorient)    
#v=pims.TiffStack(stack_path,process_func=fiji_pims_reorient)
    
#use with new version of pims when it comes out.    
#v=getPIMWormData(stack_path)
#or #whole image

v=skio.imread(stack_path)#only works in python2
v=v.transpose((0,2,1))
#v=pims.TiffStack(stack_path)#No reorrientation


#l_v=[frame[413:573+413,990:990+445] for frame in v]
#X=np.asarray(l_v)#slow and sometimes run out of memory


z_range=[min(list(zip(*marker_list))[2]),max(list(zip(*marker_list))[2])]
slices_per_neuron=10
neuron_window=np.array([75,75,35])#must be odd
window_radii=(neuron_window-1)/2#x,y,z
rel_marker_coord=(neuron_window-1)/2#coordinate of marker inside neuron_window volume
#window_radii=np.array([15,15,2])#x,y,z
r_row,r_col,r_z=window_radii


###Determine which pixels are neuron

#sigma=np.array([10,10,1.5])
#gaus=lambda x: math.exp(-1*np.linalg.norm( x.dot(1/sigma)  ))
#

##Select a subset of markers to inspect
#my_marker_list=[marker_list[1],marker_list[8]]
my_marker_list=[marker_list[5],marker_list[6]]

#raise Exception('so sorry stop here')

#my_marker_list=[marker_list[3],marker_list[4],marker_list[5]]
#my_marker_list=marker_list
#black_list=[7,13,22,23,25,26,27,28,29]
#background_list=my_marker_list[-4:]
#black_nuc=[my_marker_list[i] for i in black_list]
#[my_marker_list.remove(bn) for bn in black_nuc]

#def centerMarker(marker,X,window_radii):
#    region,local_marker=getRegionBdry(marker,window_radii,X.shape)
#    patch=contiguous_image_at_grid(X,np.roll(region,2))#np.roll moves it to z-first
#    
#    marker=np.array(marker)
#    offset_=marker-local_marker
#    old_center=marker
#    new_center=np.zeros(3)
#    denom=0
#    for loc in np.ndindex(patch.shape):
#        #np.roll(,2) for 3x1 arrays converts to x,y,z format
#        weight=gaus(np.roll(loc,2)+offset_-old_center)*patch[loc]
#        new_center+=weight*np.roll(np.array(loc),2)
#        denom+=weight
#    new_center/=denom
#    new_center=(new_center+offset_).astype('int')
##    centered_marker_list=[old_center,new_center]
#    return new_center
#



############MARKER LOOP  (whole image style)
#
#
#kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
#Blur=cv2.GaussianBlur(X,(5,5),0)
#Blur=gaussian_filter(X,(5,5,3))
#
#
#for i,m in enumerate(my_marker_list):
#    
#    m0,m1=my_marker_list
#
#    patch=getLocalImage(m0,X)
#    mid=int(len(patch)/2)+1
#    
#    ##Super simple volume extraction:
##    blur = cv2.GaussianBlur(patch,(5,5),0)#aplies blur to each image somehow
#    blur=getLocalImage(m0,Blur)
#
#    threshold_global_otsu = threshold_otsu(blur)
#    otsu=blur.copy()
#    otsu[blur<threshold_global_otsu]=0
#    otsu[blur>=threshold_global_otsu]=1
#    
#    closeP=cv2.morphologyEx(otsu,cv2.MORPH_CLOSE,kernel)
#    
#    img=otsu[mid]
#    CC=m.label(img,neighbors=8)
#    img[:,35]=3
#
############ END MARKER LOOP

def out_of_bounds(marker_ary):
#    print isinstance(args,list)
    low,high=marker_ary-window_radii,marker_ary+window_radii+1
    if not ( (low<0).any() or (high[:2]>v.frame_shape[::-1]).any() or (high[2]>len(v)) ):
#        if not :
#            if not high[2]>len(v):
        return False
    else:
        return True


def getLocalImage(marker_loc,v):
#    if out_of_bounds(marker_loc):
#        raise Exception('Marker loc + window radii goes beyond edge of image')
    low,high=marker_loc-window_radii,marker_loc+window_radii+1
    if isinstance(v,np.ndarray):
        return v[low[-1]:high[-1],low[0]:high[0],low[1]:high[1]]
    else:#is pims
        return np.array( [ _v[low[0]:high[0],low[1]:high[1]] for _v in v[low[-1]:high[-1]] ] )

#m0,m1=my_marker_list
#
#patch=getLocalImage(m0,v)
#mid=int(len(patch)/2)
#
###Super simple volume extraction:
#blur = cv2.GaussianBlur(patch,(5,5),0)#aplies blur to each image somehow
#threshold_global_otsu = threshold_otsu(blur)
#otsu=blur.copy()
#otsu[blur<threshold_global_otsu]=0
#otsu[blur>=threshold_global_otsu]=1
#
#kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
#closeP=cv2.morphologyEx(otsu,cv2.MORPH_CLOSE,kernel)
#
#img=otsu[mid]
#CC=m.label(img,neighbors=8)
#img[:,35]=3
#
#squarebysideplot(patch)
#squarebysideplot(closeP)
#
#
#
#patch1=getLocalImage(m1,v)
#mid1=int(len(patch1)/2)
###Super simple volume extraction:
#blur1 = cv2.GaussianBlur(patch1,(5,5),0)#aplies blur to each image somehow
#threshold_global_otsu1 = threshold_otsu(blur1)
#otsu1=blur1.copy()
#otsu1[blur1<threshold_global_otsu1]=0
#otsu1[blur1>=threshold_global_otsu1]=1
#kernel1=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
#closeP1=cv2.morphologyEx(otsu1,cv2.MORPH_CLOSE,kernel1)
#img1=otsu1[mid1]
#
#squarebysideplot(patch1)
#squarebysideplot(closeP1)


######CREATE OUTPUT IMAGE####
#insert=np.array([cl.transpose() for cl in closeP]).astype(np.uint8)
#insert*=255
#insert1=np.array([cl.transpose() for cl in closeP1]).astype(np.uint8)
#insert1*=255
#
#
##Build an output image.
#Y=np.zeros((len(v),v.frame_shape[0],v.frame_shape[1]),dtype=np.uint8)
#low,high=m0-window_radii,m0+window_radii+1
#Y[low[2]:high[2],low[1]:high[1],low[0]:high[0]]=insert
#
#low,high=m1-window_radii,m1+window_radii+1
#Y[low[2]:high[2],low[1]:high[1],low[0]:high[0]]=insert1
#
##Y=Y[:,::-1,:]
#
##Works according to imageJ
#imsave(WormConfig['working_directory']+'/'+'150226 Hi-res_40x W5 uint8 seg local transpose.tif',Y,compress=1)

##########END CREATE OUTPUT IMAGE######

#lb=[]#v[109] is the problem
#for _v in v[95:high[-1]]:
#    _v
#    lb.append(k[low[0]:high[0],low[1]:high[1]])
    


def inspect_marker(marker_idx,marker_list=marker_list):
    m0=marker_list[marker_idx]
    patch=getLocalImage(m0,v)
    #    mid=int(len(patch)/2)
    
    ##Super simple volume extraction:
    blur = cv2.GaussianBlur(patch,(5,5),0)#aplies blur to each image somehow
    threshold_global_otsu = threshold_otsu(blur)
    otsu=blur.copy()
    otsu[blur<threshold_global_otsu]=0
    otsu[blur>=threshold_global_otsu]=1
    
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    closeP=cv2.morphologyEx(otsu,cv2.MORPH_CLOSE,kernel)
    
    #    img=otsu[mid]
    #    CC=m.label(img,neighbors=8)
    #    img[:,35]=3
    
    squarebysideplot(patch,[str(marker_idx)])
    squarebysideplot(closeP,[str(marker_idx)])
    return patch,closeP



_p=inspect_marker(19)

#np.save(WormConfig['working directory']+'/'+WormConfig['data set identifier'],patch)

patch,binary=inspect_marker(11)
#CC=m.label(binary,neighbors=8)

#R=m.regionprops(CC[13])


from sklearn.cluster import MeanShift, estimate_bandwidth

#%%

#img=patch[18]
#img=binary[18]

#px,py=np.where(img==1)
#fea=np.vstack([px,py]).transpose()
pos=np.where(binary==1)
fea=np.vstack(pos).transpose()

#px,py=np.indices(img.shape)
#fea=np.vstack([px.ravel(),py.ravel(),img.ravel()])

##data whitening
#from scipy.linalg import fractional_matrix_power
#u=np.mean(fea,axis=1).reshape(3,1)
#fea=fea-u#centering
#Cov=np.dot(fea,fea.transpose())
#rootCov=fractional_matrix_power(Cov,0.5)
#negrootCov=fractional_matrix_power(Cov,-0.5)
#wfea=np.dot(negrootCov,fea)#whitening transform

bandwidth = estimate_bandwidth(fea,quantile=0.25)#less scalable than algorithm. estimate.
print 'bandwidth is ',bandwidth
#ms = MeanShift(bandwidth=.5, bin_seeding=True)
#ms = MeanShift(bandwidth=.05)
ms = MeanShift(bandwidth=bandwidth,bin_seeding=True)
ms.fit(fea)#expects samples x features
labels = ms.labels_
cluster_centers = ms.cluster_centers_
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
print 'there are ',n_clusters_,' clusters'


dst=np.zeros(binary.shape)
for l in labels_unique:
    lab_l=[labels==l]
    pos_l=[p[lab_l] for p in pos]
    dst[pos_l]=l+1
    
squarebysideplot(dst,colormaps=['jet'])

#########

#%%


from skimage.morphology import medial_axis
from scipy.ndimage.morphology import distance_transform_bf,distance_transform_cdt

data=np.array([[0,0,0,0,0,0,0],[0,1,1,1,1,1,0],[0,1,1,1,1,1,0],[0,1,1,1,1,1,0],[0,1,1,1,1,1,0],[0,1,1,1,1,1,0],[0,0,0,0,0,0,0]])
zed=np.zeros((7,7))
data3=np.dstack([zed,data,data,data,data,data,zed])

distance_transform_bf(data3,'euclidean',sampling=[1,1,2])#sampling is cost for moving in that direction

#skel, distance = medial_axis(data3, return_distance=True)

dt=distance_transform_bf(binary,'euclidean',sampling=[3,1,1])

fig,axes=squarebysideplot(dt)

from skimage.feature import peak_local_max

#idx=peak_local_max(dt,min_distance=0,exclude_border=False)
#indices_=zip(*idx)
thper=np.percentile(dt.ravel(),95)
indices_=np.where(dt>thper)

indices=[np.array(ind) for ind in indices_]

z_max=max(indices[0])

for z in range(z_max+1):
        y=indices[1].take(np.where( (indices[0]==z) ))
        x=indices[2].take(np.where( (indices[0]==z) ))
        axes[z].scatter(x,y,c='b',marker='d',s=7)
fig.canvas.draw()





#from scipy.ndimage.measurements import extrema #whole image
#et=extrema(dt)


#imsave(WormConfig['working_directory']+'/'+'patch1.tif',patch1,compress=1)
#x=v[79]
#y=Y[79]
#xloc=x[low[0]:high[0],low[1]:high[1]]#as in get local image
#yloc=y[low[1]:high[1],low[0]:high[0]]#as in def of Y
#ploc=patch1[17]
#
#sidebysideplot([xloc,yloc,ploc])
#
#
#
###why is ploc.sum() different from xloc.sum() ??
#
#marker_loc=m1
#low,high=marker_loc-window_radii,marker_loc+window_radii
#hatpatch1= np.array( [ _X[low[0]:high[0],low[1]:high[1]] for _X in v[low[-1]:high[-1]] ] )
#hatploc1=hatpatch1[17]







#maybe write it here so that it saves each slice individually,
    #instead of having to do that manually in imagej

#im=closeP1[5]
#R=m.regionprops(im)



#for region in m.regionprops(img):#labels with value 0 are ignored.
#    print region.area
#R=m.regionprops(img)
#r=R[0]

#def getNucPixels(marker_list,X,window_radii):
##    labels=[]
#    marker_index=[[],[],[]]
#    each_patch=[]
#    each_nuc=[]
#    for i,marker in enumerate(centered_marker_list):
#        region,offset=getRegionBdry(marker,window_radii,X.shape)
#        #note to future self: np.roll(,2) works here because each entry in region is really
#        #a 2 dim element, so to convert back would be np.roll(,4)
#        patch=contiguous_image_at_grid(X,np.roll(region,2))#np.roll moves it to z-first
#        each_patch.append(patch)
#        
#        #Decide which pixels count as nuc
#        th=np.mean(patch)
#        nuc_pix=patch>th
#        [ind.extend(list(x)) for ind,x in zip(marker_index,nuc_pix.nonzero())]
#        each_nuc.append(np.array([list(x) for x in nuc_pix.nonzero()]))
##        labels.extend([i for x in nuc_pix.nonzero()[0]])    
##    labels=np.array(labels)
##    indices=[np.array(m_idx) for m_idx in marker_index]
#
#    return each_nuc,each_patch
#
#centered_marker_list=[centerMarker(m,X,window_radii) for m in my_marker_list]
#each_nuc,each_patch=getNucPixels(centered_marker_list,X,window_radii)
#nu,pa=each_nuc[0],each_patch[0]
#
#nu_sum=0
#count=0
#for i in range(len(nu[0])):
#    nu_sum+=pa[nu[0][i]][nu[1][i]][nu[2][i]]
#    count+=1
#nu_mean=float(nu_sum)/count
##pa[nu]
#
#def getBackgroundPixels(bk_marker_list,X,window_radii):
#    marker_index=[[],[],[]]
#    each_patch=[]
#    each_nuc=[]        
#    for i,marker in enumerate(bk_marker_list):
#        region,offset=getRegionBdry(marker,window_radii,X.shape)
#        #note to future self: np.roll(,2) works here because each entry in region is really
#        #a 2 dim element, so to convert back would be np.roll(,4)
#        patch=contiguous_image_at_grid(X,np.roll(region,2))#np.roll moves it to z-first
#        each_patch.append(patch)
#        #Decide which pixels count as nuc
#        #        th=np.mean(patch)
#        th=np.min(patch)-1
#        nuc_pix=patch>th
#        [ind.extend(list(x)) for ind,x in zip(marker_index,nuc_pix.nonzero())]
#        each_nuc.append(np.array([list(x) for x in nuc_pix.nonzero()]))
#    return each_nuc,each_patch
#        
#bk_nucs,bk_patches=getBackgroundPixels(background_list,X,window_radii)



#Now get timeseries
#foldername="C:\\Users\\melocaladmin\\Documents\\WormPics\\141226_CM_Pan_GCamp_Vol1_Stacks\\"
#foldername='./../WormPics\\141226_CM_Pan_GCamp_Vol1_Stacks'
#time_stack_file="c_141226_ZIM294 Vol1_L4_O20X_F630x75um_Vol_P6144x500S28_g2_stack1of7.tiff"
#subfolders=['stack1split','stack2split','stack3split','stack4split','stack5split','stack6split','stack7split']
#subfolders=['\stack1split','\stack2split','\stack3split','\stack4split','\stack5split','\stack6split','\stack7split']
#v=time_stack_read()#Doesn't work! Stack too big

#all_files=[]    
#for sub in subfolders:
#    filenames=ls(foldername+sub)
#    all_files.extend(filenames)
#sort_nicely(all_files)#alphanumeric sorting
#
#sz=all_files.__len__()
#
#t_dim=50*7
#z_dim=10


#Convert indicies to new grid
#Assume same volume of view








#indicies=np.resize(np.linspace(0,sz-1,sz).astype('int'),(t_dim,z_dim))
#total_light=[]

#f=open('total_volume_integral','w')
#for idx in indicies:
#    lgt=0    
#    for i in idx:
##        print i
#        X=cv2.imread(all_files[i],0)
#        lgt+=np.sum(X,dtype=np.uint16)        
#    total_light.append(lgt)
#    f.write('%d %d \n' % (idx[0]/10,lgt))
#
f.close()


#data=np.loadtxt(foldername + 'total_volume_integral')
#(time,Light)=zip(*data)       
##Light=np.array(total_light).astype(np.float32)
#percent_change_Light=100*(Light-np.mean(Light))/np.mean(Light)
#
#time=np.linspace(0,120,t_dim)
#plt.plot(time,percent_change_Light)
#plt.xlabel('Time (s)')
#plt.ylabel('Percent Intensity Change (relative to Average)')
#plt.title('Total Volume Intensity Change over Time')
#plt.show(block=False)




#    
#    for im,mark in zip(each_patch,each_nuc):
#        nucPatchPlot(im,mark)

