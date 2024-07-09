# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 22:36:27 2016

@author: csnyder
"""


import numpy as np
#from WormBox.BaseClasses import tEnvironment as Env
from WormBox.BaseClasses import tWorm
from WormBox.VisClasses import MultiChannelWorm
from WormScene import WormScene,Worm4DScene
from Display import Display
from WormBox.ioworm import smakedirs
from functools import partial


####10mM###
#ini1='/work/03176/csnyder/Corral/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/bin_info.ini'
####img1 file denotes which roi we're dealing with####
#img1='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/pos-3_248_233_len23_157_251/Correctedpos-3_248_233_len23_157_251.tif'
#img1='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/pos-3_106_361_len23_173_191/Correctedpos-3_106_361_len23_173_191.tif'
#img1='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/pos0_49_439_len17_107_203/Correctedpos0_49_439_len17_107_203.tif'
#img1='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/pos0_69_637_len17_107_203/Correctedpos0_69_637_len17_107_203.tif'
#img1='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/pos0_142_782_len17_223_203/Correctedpos0_142_782_len17_223_203.tif'
#img1='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/pos0_318_880_len17_151_271/Correctedpos0_318_880_len17_151_271.tif'
#img1='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/pos0_62_1079_len17_319_233/Correctedpos0_62_1079_len17_319_233.tif'
#img1='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/FOV8/NoCorrectionNeeded_FOV8.tif'
#img1='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/FOV9/NoCorrectionNeeded_FOV9.tif'
#img1='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/pos-2_335_1722_len15_63_113/Correctedpos-2_335_1722_len15_63_113.tif'
#img1='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1849_Vol_NaCl_10mM_30s_4min_W1/Results/pos-2_244_1853_len25_149_203/Correctedpos-2_244_1853_len25_149_203.tif'

####10mM###
#ini1='/work/03176/csnyder/Corral/Ki-Confocal-2015/151214_1930_Vol_NaCl_10mM_30s_4min_W2/bin_info.ini'
#img1='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151214_1930_Vol_NaCl_10mM_30s_4min_W2/Results/pos-2_114_236_len19_263_387/Correctedpos-2_114_236_len19_263_387.tif'


####50mM###
ini1='/work/03176/csnyder/Corral/Ki-Confocal-2015/151217_1915_Vol_NaCl_50mM_30s_4min_W1/bin_info.ini'
img1='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151217_1915_Vol_NaCl_50mM_30s_4min_W1/Results/pos-3_87_182_len23_305_297/Correctedpos-3_87_182_len23_305_297.tif'

####50mM###
#ini1='/work/03176/csnyder/Corral/Ki-Confocal-2015/151217_2109_Vol_NaCl_50mM_30s_4min_W5/bin_info.ini'
#img1='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151217_2109_Vol_NaCl_50mM_30s_4min_W5/Results/pos0_144_95_len19_177_211/Correctedpos0_144_95_len19_177_211.tif'

#ini1='/work/03176/csnyder/Corral/Ki-Confocal-2015/151217_2010_Vol_NaCl_50mM_30s_4min_W3/bin_info.ini'
#img1='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151217_2010_Vol_NaCl_50mM_30s_4min_W3/Results/pos-3_172_259_len25_233_311/Correctedpos-3_172_259_len25_233_311.tif'


##Did not work:
#ini1='/work/03176/csnyder/Corral/Ki-Confocal-2015/151215_2047_Vol_NaCl_10mM_30s_4min_W2/bin_info.ini'
#img1='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151210_1640_Vol_NaCl_10mM_30s_4min_W2/Results/pos0_293_329_len7_95_115/Correctedpos0_293_329_len7_95_115.tif'

#ini1='/work/03176/csnyder/Corral/Ki-Confocal-2015/151125_1814_Vol_NaCl_10mM_30s_4min_W2/bin_info.ini'
#img1='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151125_1814_Vol_NaCl_10mM_30s_4min_W2/Results/pos-1_225_-5_len19_175_251/Corrected_with_gaps.sima/volume.tif'
#cache_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151125_1814_Vol_NaCl_10mM_30s_4min_W2/Results/pos-1_225_-5_len19_175_251/Cache/'
#results_dir='/work/03176/csnyder/Corral/Snyder/WormPics/Ki-Confocal-2015/151125_1814_Vol_NaCl_10mM_30s_4min_W2/Results/pos-1_225_-5_len19_175_251/Results/'

p=img1.rfind('/')
folder=img1[:p]
cache_dir=folder+'/Cache/'
results_dir=folder+'/Results/'
smakedirs(cache_dir,results_dir)


def max_box_average(duration,array):
    rad=int(duration//2)
    L=len(array)    
    current_max=np.ones(array.shape[1:])*-np.inf
    for t in range(array.shape[0]):###Takes a really long time, nonparallel
        a=max(0,t-rad)
        b=min(t+rad,L)
        current_ave = np.mean(np.array(array[a:b]),axis=0)
        #"maximum" is elementwise max
        current_max=np.maximum(current_max,current_ave)
    return current_max
box30= partial(max_box_average,30)


#def std_dev(array):
#    return np.std(array,axis=0)

def smooth(duration,array):
    rad=int(duration//2)
    L=len(array)    
    out=[]
    for t in range(array.shape[0]):###Takes a really long time, nonparallel
        a=max(0,t-rad)
        b=min(t+rad,L)
        current_ave = np.mean(np.array(array[a:b]),axis=0)
        out.append(current_ave[None,:])
    return np.vstack(out)
smoo30=partial(smooth,30)
smoo60=partial(smooth,60)

def norm_dev(array):
    return (np.std(array/smoo60(array),axis=0)*10)**4


um_window_shape=np.array([10.0,10.0,10.0])
nm_window_shape=1000*um_window_shape



from skimage import io
X=io.imread(img1)
w1=tWorm(X,ini1)
nm_image_shape=w1.nm_image_shape


worm=MultiChannelWorm(nm_window_shape, nm_image_shape)
worm.cache_dir=cache_dir
worm.results_dir=results_dir




worm.add_channel_directly('vol',w1)


vol=worm['vol']

worm.add_channel_derived_from('vol',with_process=box30, named='mba30')
#worm.add_channel_derived_from('vol',with_process=std_dev, named='std')
#worm.add_channel_derived_from('vol',with_process=norm_dev, named='nstd')


worm.segmentation_channel_is('mba30')
worm.extract_time_series_data_for('vol')

gui=Display()
gui.worm=worm
#gui.add_scene('Global_Std',WormScene() )
gui.add_scene('Global_Max_Box_Average',WormScene() )
gui.add_scene('Max_Box_Average',WormScene(is_local_view=True) )
gui.add_scene('Time_Series', Worm4DScene(is_local_view=True) )

gui.hash_volume=dict({
#    'Global_Std':'nstd',
#    'Global_Std':'std',
    'Global_Max_Box_Average':'mba30',
    'Max_Box_Average':'mba30',
    'Time_Series':'vol',
    })

gui.start()



def f():
    chan=gui.worm['mba30']
    d=5#downsample
    scale=chan.nm_voxel_shape
    dar=np.array(chan.X[:,::d,::d])
    izi,iyi,ixi=np.indices(dar.shape)
    from mayavi import mlab
    text_scale=5*np.array([1,1,1])
    
    mlab.figure()
    s=mlab.pipeline.scalar_field(izi*chan.spacing[0],d*iyi*chan.spacing[1],d*ixi,dar*chan.spacing[2])
    v=mlab.pipeline.volume(s)
    key,neu=gui.worm.segments.items()[0]
    for key,neu in gui.worm.segments.items():
        p=neu['marker']//scale * chan.spacing
        mlab.text3d(x=p[0],y=p[1],z=p[2],text=key,orient_to_camera=True,line_width=10,scale=text_scale)
        nodes=mlab.points3d(p[0],p[1],p[2],name='new marker',color=(1,0,0))
        nodes.glyph.glyph.scale_factor=5
    #    nodes.glyph.glyph.scale_factor=1


stophere




a=gui.analyzer
a.calculate_all_intensities_from_segments()
a._prelim_analysis()
a.plot_time_series()
a.plot_correlation()
a.write_csv()
a.plot_fancy_time_series()



import matplotlib.pyplot as plt
plt.tight_layout(pad=0.5,w_pad=0.5,h_pad=0.6)


f()





