# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 08:33:53 2015

@author: melocaladmin
"""


'''
Watershed segmentation
=========

This program demonstrates the watershed segmentation algorithm
in OpenCV: watershed().

Usage
-----
watershed.py [image filename]

Keys
----
  1-7   - switch marker color
  SPACE - update segmentation
  r     - reset
  a     - toggle autoupdate
  ESC   - exit
  d     - zoom in 
'''
'''
Developer notes:
Currently I'm pretty sure it only works if you feed the file name
Could be interesting to try other methods besides clean8()
Filename must be to a 16bit image
Does support scrolling to the end of the page yet.

'''





import cv2
import numpy as np
from ioworm import getPIMWormData
from itertools import repeat
import matplotlib.pyplot as plt
from WormPlot import sidebysideplot
from formatworm import clean8


class Subidx:
    def __init__(self,img):
        nrows,ncols=img.shape[:2]
        self.h,self.w=nrows,ncols
        [self.left,self.top,self.right,self.bottom]=[0,0,ncols,nrows]
#        self.wholeImage=img
#        self.refresh_sub_img()
        self.rel_mouse_pos = (0.0,0.0)
        self.abs_mouse_pos=(0.0,0.0)
    def zoom_in(self):
        #reduce margins on all sides by p=30%
        (x,y)=self.rel_mouse_pos        
        p=0.3
        self.left+=int(p*(x-self.left))
        self.right-=int(p*(self.right-x))
        self.top+=int(p*(y-self.top))
        self.bottom-=int(p*(self.bottom-y))
#        self.viewImage=self.wholeImage[self.top:self.bottom,self.left:self.right]
        
    def add_offset(self, pt ):
        return (int(pt[0]+self.left),int(pt[1]+self.top))    
    
#    def refresh_sub_img(self):
#        self.viewImage=self.wholeImage[self.top:self.bottom,self.left:self.right]

    def reset_zoom(self):
        [self.left,self.top,self.right,self.bottom]=[0.0,0.0,self.w,self.h]
#        self.refresh_sub_img()

    def get_sub_image_of_this(self,an_img):
         return an_img[self.top:self.bottom,self.left:self.right]
    
#    def reset_subimage(self):
#        nrows,ncols,ncolors=self.wholeImage.shape
#        [self.left,self.top,self.right,self.bottom]=[0.0,0.0,ncols,nrows]
#        self.refresh_sub_img()

class Sketcher3D:
    def __init__(self, windowname, dests, colors_func,imgscale=0.3,is3d=False,subreg=-1,mid=-1):
        self.cur_middle=mid
        self.prev_pt = None
        self.windowname = windowname
        self.dests = dests
        self.colors_func = colors_func
        self.dirty = False
        self.mode3d=is3d
        self.subregion=subreg
        self.scale=imgscale
        self.update_window()
        
    def update_window(self):
        nrows,ncols=self.subregion.h,self.subregion.w        
        scale=self.scale
        cv2.namedWindow(self.windowname,cv2.cv.CV_WINDOW_NORMAL)#allows resize manual
        cv2.moveWindow(self.windowname,0,0)
        cv2.resizeWindow(self.windowname,int(scale*ncols),int(scale*nrows))
        self.show()
        cv2.setMouseCallback(self.windowname, self.on_mouse)
        self.mouse_pos=(0,0)
        
    def show(self):
#        print self.dests[0]
#        print 'd'
#        print self.cur_middle
        q=self.subregion.get_sub_image_of_this(self.dests[0][self.cur_middle])
#        print q
        cv2.imshow(self.windowname, q)

    def on_mouse(self, event, x, y, flags, param):
        pt_rel=(x,y)
        self.subregion.rel_mouse_pos=pt_rel
        pt =self.subregion.add_offset( pt_rel )
        self.subregion.abs_mouse_pos=pt
        if event == cv2.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        if self.prev_pt and flags & cv2.EVENT_FLAG_LBUTTON:
            
            dst=self.dests[0][self.cur_middle]
            color=self.colors_func()[0]
            cv2.line(dst, self.prev_pt, pt, color, 5)
            
            dst=self.dests[1][self.cur_middle]
            color=self.colors_func()[1]
            cv2.line(dst, self.prev_pt, pt, color, 5)
            
#            for dst, color in zip([self.dests[0][self.cur_middle],self.dests[1][self.cur_middle]], self.colors_func()):
##                print 'del'                
##                print dst
##                print 'del'
##                print color
##                print 'del'
#                 cv2.line(dst, self.prev_pt, pt, color, 5)
#            print map(np.max,self.dests[1])
            self.dirty = True
            self.prev_pt = pt
            self.show()
        else:
            self.prev_pt = None

class sidebyside:
    def __init__(self,mid=-1,numplots=5):
        self.cur_middle=mid
        self.fig=-1
        self.axes=-1
        self.n_plots=numplots
        self.init_left_right()
        self.rel_cur_middle=self.cur_middle-self.l        
#    def update_and_plot(self):
##         print "update and plot"
#         self.init_left_right(self.cur_middle)
#         self.set_cur_plots()
#         self.plot_cur_plots()
#         print "update_and_plot"
#         print self.all_plots
#         print self.all_plots[0][0][0]
    def init_left_right(self):
        self.l=max(0,self.cur_middle-int(self.n_plots/2))
        self.r=self.l+self.n_plots-1
    def set_cur_plots(self,v,switch=1):
        self.cur_plots=[]
        for img in v:
            if switch==1:
                temp=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            elif switch==0:
                temp=img
            self.cur_plots.append(temp)
        return self.cur_plots
    def set_mid_plot(self,img):
        self.cur_plots[self.rel_cur_middle]=cv2.cvtColor(img,cv2.cv.CV_RGB2BGR)
    def get_cur_list_of_item(self,item):
        itemlist=[]
        for i in range(self.l,self.r+1):
            itemlist.append(item[i])
        return itemlist
    def init_plots(self):
        self.fig,self.axes=sidebysideplot(self.cur_plots,self.get_titles(),colormap='standard')
    def plot_cur_plots(self):
        for ax,img,title in zip(self.axes,self.cur_plots,self.get_titles()):
            if self.n_plots==1:
                ax=ax[0]
            ax.clear()
            ax.imshow(img)
#            ax.draw
            ax.set_title(title)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
    def plot_middle_plot(self):
        ax=self.axes[self.rel_cur_middle]
        title=self.get_titles()[self.rel_cur_middle]
        img=self.cur_plots[self.rel_cur_middle]
        ax.clear()
        ax.imshow(img)
        ax.draw
        ax.set_title(title)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    def get_titles(self):
        return map(str,range(self.l,self.r+1))
        
#    def set_cur_plots(self,v):
##        print "change_all_plots"
##        print v        
#        self.cur_plots=self.change_to_BGR(v)
    def change_to_BGR(self,v):
        lst=[]
        for img in v:
            lst.append(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        return lst
        
class Seg3D:
    def __init__(self, fn):
        
        self.parse_input(fn)
        self.n_plots=3
        self.n_plots=min(self.v.__len__(),self.n_plots)
        self.cur_middle=int(self.z/2)
        self.watershedplot=sidebyside(mid=self.cur_middle,numplots=self.n_plots)
        self.convert_data()
        self.vis_copy=map(np.copy,self.X)
        self.watershedplot.set_cur_plots(self.X[self.watershedplot.l:self.watershedplot.r+1])
        self.watershedplot.init_plots()
        self.init_subregion()
        self.init_markers()
        self.generate_colors()
#        self.colorlist=[' unknown',' red',' green',' yellow',' darkblue',' magenta',' teal']    
        self.auto_update = True
        self.sketch = Sketcher3D('img', [self.markers_vis, self.markers], self.get_colors,is3d=self.mode3d,subreg=self.subregion,mid=self.cur_middle)
#        self.vis_copy=self.X[:]#hard copy
#    def watershed_prep(self,img16u):
#        clean=clean8(img16u)
#        dist=
    def update_only_middle_plot_with(self,img):
        self.watershedplot.set_mid_plot(img)
        self.watershedplot.plot_middle_plot()
    def get_colors(self):
        return map(int, self.colors[self.cur_marker]), self.cur_marker
    def watershed(self):
        print("watershed was called")
        m=self.markers[self.cur_middle].copy()
        img=self.X[self.cur_middle]        
        cv2.watershed(img,m)
        overlay = self.colors[np.maximum(m, 0)]
        vis = cv2.addWeighted(img, 0.5, overlay, 0.5, 0.0, dtype=cv2.CV_8UC3)
        sub_vis=self.subregion.get_sub_image_of_this(vis)
        self.vis_copy[self.cur_middle]=sub_vis
        self.update_only_middle_plot_with(sub_vis)
                    
    def generate_colors(self):
        self.colors = np.int32( list(np.ndindex(2, 2, 2)) ) * 255
        self.bgr_colors=self.colors.copy()
        self.bgr_colors[:,0],self.bgr_colors[:,2]=self.colors[:,2],self.colors[:,0]
    def parse_input(self,fn):
        if isinstance(fn,str):  
            self.v=getPIMWormData(fn)
            if self.v.__len__()==1:
                self.mode3d=False
                self.z=1
            elif self.v.__len__()>1:
                self.mode3d=True
                self.z=self.v.__len__()                
        elif isinstance(fn,np.ndarray):
            self.v=fn.copy()
        else:#assume PIMstack was passed
            self.v=fn
            self.mode3d=True
    def convert_data(self):#b&w to color and pims to list, and 16 to 8bit
        self.X=[]        
#        for img in self.v:
#            self.X.append(cv2.cvtColor(clean8(img),cv2.cv.CV_GRAY2RGB))

        self.X=list(repeat(-1,self.z))
        self.in_memory=[]
        for i in range(self.watershedplot.l,self.watershedplot.r+1):
            self.X[i]=cv2.cvtColor(clean8(self.v[i]),cv2.cv.CV_GRAY2RGB)#only set the ones which are needed
            self.in_memory.append(i)
    def init_subregion(self):
#            print self.X
            self.h, self.w = self.X[self.cur_middle].shape[:2]
            self.subregion=Subidx(self.X[self.cur_middle])
            
    def init_markers(self):
        marker=np.zeros((self.h, self.w), np.int32)
#        self.markers=list(repeat(marker,self.z))
#        self.markers=[marker]*self.z        
        self.markers=[]
        for i in range(self.z):
            self.markers.append(marker.copy())
        self.cur_marker=1
        self.markers_vis=map(np.copy,self.X)
#        print self.markers_vis
    def refresh_plots(self):
        self.watershed()
        self.sketch.show()
    def zoom_in(self):
        self.subregion.zoom_in()
        self.sketch.show()
#        self.watershedplot.set_cur_plots(self.find_cur_portion(self.vis_copy))
        self.watershedplot.set_cur_plots(map(self.subregion.get_sub_image_of_this,self.watershedplot.cur_plots),switch=0)
        self.watershedplot.plot_cur_plots()
    
    def zoom_out(self):
        self.subregion.reset_zoom()
        self.sketch.show()
        self.watershedplot.set_cur_plots(self.find_cur_portion(self.vis_copy))
        self.watershedplot.plot_cur_plots()
    def find_cur_portion(self,list_of_images):
        short_list=self.watershedplot.get_cur_list_of_item(list_of_images)
        return map(self.subregion.get_sub_image_of_this,short_list)
    def check_bounds(self,idx):
        if idx<0:
            print "moved as far left as possible already"
            return 0
        elif idx==self.z:
            print "moved as far right as possible already"
            return 0
        else:
            return 1
             
    def adjust_params_by(self,delta):
        self.cur_middle+=delta
        self.watershedplot.cur_middle+=delta
        self.watershedplot.l+=delta
        self.watershedplot.r+=delta
        self.sketch.cur_middle+=delta
    def move_left(self):
        new_idx=self.watershedplot.l-1
        if self.check_bounds(new_idx):
            self.adjust_params_by(-1)
            print('moved left')
            if new_idx in self.in_memory:
                print('in memory already')
                self.watershedplot.set_cur_plots(self.find_cur_portion(self.vis_copy))
                self.watershedplot.plot_cur_plots()
                self.sketch.show()
            else:#not yet loaded into memory
                self.X[new_idx]=cv2.cvtColor(clean8(self.v[new_idx]),cv2.cv.CV_GRAY2RGB)#load from PIMs
                self.vis_copy[new_idx]=self.X[new_idx].copy()
                self.markers_vis[new_idx]=self.vis_copy[new_idx].copy()
                self.watershedplot.set_cur_plots(self.find_cur_portion(self.vis_copy))
                self.watershedplot.plot_cur_plots()
                self.sketch.show()
    def move_right(self):
        new_idx=self.watershedplot.r+1
        if self.check_bounds(new_idx):
            self.adjust_params_by(1)
            print('moved right')
            if new_idx in self.in_memory:
                print('in memory already')
                self.watershedplot.set_cur_plots(self.find_cur_portion(self.vis_copy))
                self.watershedplot.plot_cur_plots()
                self.sketch.show()
            else:#not yet loaded into memory
                self.X[new_idx]=cv2.cvtColor(clean8(self.v[new_idx]),cv2.cv.CV_GRAY2RGB)#load from PIMs
                self.vis_copy[new_idx]=self.X[new_idx].copy()
                self.markers_vis[new_idx]=self.vis_copy[new_idx].copy()
                self.watershedplot.set_cur_plots(self.find_cur_portion(self.vis_copy))
                self.watershedplot.plot_cur_plots()
                self.sketch.show()
    def run(self):
        while True:
            ch = 0xFF & cv2.waitKey(300)
            if ch == 27:
                break
            if ch >= ord('1') and ch <= ord('7'):
                self.cur_marker = ch - ord('0')
                print 'marker: ', self.cur_marker,self.colors[int(self.cur_marker)]
            if ch == ord(' ') or (self.sketch.dirty and self.auto_update):
                self.refresh_plots()
                self.sketch.dirty = False
            if ch in [ord('a'), ord('A')]:
                self.auto_update = not self.auto_update
                print 'auto_update if', ['off', 'on'][self.auto_update]
            if ch in [ord('r'), ord('R')]:
                self.init_markers()
                self.sketch.dests=[self.markers_vis, self.markers]
                self.subregion.reset_zoom()
                self.sketch.show()
                self.watershedplot.set_cur_plots(map(self.subregion.get_sub_image_of_this,self.watershedplot.cur_plots),switch=0)
                self.watershedplot.plot_cur_plots()
            if ch in [ord('d'),ord('D')]:
                print "d was struck"
                self.zoom_in()
            if ch in [ord('f'),ord('F')]:
                self.zoom_out()
#                self.subregion.reset_zoom()
#                self.refresh_plots()
            if ch in [ord('w'),ord('W')]:
                self.move_left()
            if ch in [ord('e'),ord('E')]:
                self.move_right()
        cv2.destroyAllWindows()
        plt.close('all')
#        cv2.namedWindow('watershed',cv2.cv.CV_WINDOW_NORMAL)#allows resize manual
#        cv2.moveWindow('watershed',0,100)
#        nrows,ncols,ncolors=self.img.shape    
#        cv2.resizeWindow('watershed',int(self.scale*ncols),int(self.scale*nrows))
        
        
#if __name__ == '__main__':
#    import sys
pcfoldername="C:\Users\melocaladmin\My Documents\WormPics"
pcdataname="\\20140908 Confocal ZIM294 Step Size Stacks Split\\14-09-08 Confocal ZIM294 L4 W1 H 1.5um Stack Split\\14-09-08 Confocal ZIM294 L4 W1 H 1.5um Stack Split_7.tif"
path=pcfoldername+pcdataname
#    try: fn = sys.argv[1]
##    except: fn = '../cpp/fruits.jpg'
#    except: fn=path
#    print __doc__
#    WaterApp(fn).run()
    
pcfoldername="C:\Users\melocaladmin\My Documents\WormPics"
pcdataname="\\20140908 Confocal ZIM294 Step Size Stacks\\14-09-08 Confocal ZIM294 L4 W1 H 1.5um Stack.tif"
fn=pcfoldername+pcdataname


a=Seg3D(fn)
a.run()
#b=Seg3D(fn)
    
#te=cv2.imread(path)
#v=cv2.cvtColor(te,cv2.cv.CV_GRAY2RGB)


    