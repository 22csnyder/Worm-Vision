#!/usr/bin/env python

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
  
  
Python 3 compatible
'''
import numpy as np
#from opencv2 import cv2#for ubuntu
import cv2#win py2.7 opencv 3.0.0beta

#from common import Sketcher
class Subimage:
    def __init__(self,img):
        nrows,ncols,ncolors=img.shape
        [self.left,self.top,self.right,self.bottom]=[0.0,0.0,ncols,nrows]
        self.wholeImage=img
        self.refresh_sub_img()
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
        self.viewImage=self.wholeImage[self.top:self.bottom,self.left:self.right]
        
    def add_offset(self, pt ):
        return (int(pt[0]+self.left),int(pt[1]+self.top))    
    
    def refresh_sub_img(self):
        self.viewImage=self.wholeImage[self.top:self.bottom,self.left:self.right]

    def reset_zoom(self):
        nrows,ncols,ncolors=self.wholeImage.shape
        [self.left,self.top,self.right,self.bottom]=[0.0,0.0,ncols,nrows]
        self.refresh_sub_img()
    def get_sub_image_of_this(self,an_img):
         return an_img[self.top:self.bottom,self.left:self.right]
    
    def reset_subimage(self):
        nrows,ncols,ncolors=self.wholeImage.shape
        [self.left,self.top,self.right,self.bottom]=[0.0,0.0,ncols,nrows]
        self.refresh_sub_img()
         
class my_Sketcher:
    def __init__(self, windowname, dests, colors_func,imgscale=0.3,mid=-1):
        
        self.prev_pt = None
        self.windowname = windowname
        self.dests = dests
        self.colors_func = colors_func
        self.dirty = False
        #init position
        nrows,ncols,ncolors=self.dests[0].shape
        self.subimage=Subimage(self.dests[0])
        scale=imgscale
        cv2.namedWindow(self.windowname,cv2.WINDOW_NORMAL)#allows resize manual
        cv2.moveWindow(self.windowname,0,0)
        cv2.resizeWindow(self.windowname,int(scale*ncols),int(scale*nrows))
        self.show()
        cv2.setMouseCallback(self.windowname, self.on_mouse)
        self.mouse_pos=(0,0)
        
    def show(self):
        cv2.imshow(self.windowname, self.subimage.viewImage)

    def on_mouse(self, event, x, y, flags, param):
        pt_rel=(x,y)
        self.subimage.rel_mouse_pos=pt_rel
        pt =self.subimage.add_offset( pt_rel )
        self.subimage.abs_mouse_pos=pt
        if event == cv2.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        if self.prev_pt and flags & cv2.EVENT_FLAG_LBUTTON:
            for dst, color in zip(self.dests, self.colors_func()):
                cv2.line(dst, self.prev_pt, pt, color, 5)
            self.dirty = True
            self.prev_pt = pt
            self.show()
        else:
            self.prev_pt = None
        



class WaterApp:
    def __init__(self, fn):
        if isinstance(fn,str):
            src = cv2.imread(fn)
            if src is None:
                print("Error: image could not be read")
        elif isinstance(fn,np.ndarray):
            src=fn.copy()
        else:
            print("Input specified is neither a file path nor an image matrix")
        self.colorlist=[' unknown',' red',' green',' yellow',' darkblue',' magenta',' teal']
        self.img=cv2.cvtColor(src,cv2.COLOR_GRAY2RGB)    
#        print self.img
        h, w = self.img.shape[:2]
#        self.subimg=Subimage(self.img)       
        self.markers = np.zeros((h, w), np.int32)
        self.markers_vis = self.img.copy()
        self.cur_marker = 1
        self.colors = np.int32( list(np.ndindex(2, 2, 2)) ) * 255

        self.auto_update = True
        self.scale=0.3
        self.sketch = my_Sketcher('img', [self.markers_vis, self.markers], self.get_colors,imgscale=self.scale)
        cv2.namedWindow('watershed',cv2.WINDOW_NORMAL)#allows resize manual
        cv2.moveWindow('watershed',0,100)
        nrows,ncols,ncolors=self.img.shape    
        cv2.resizeWindow('watershed',int(self.scale*ncols),int(self.scale*nrows))        

    def get_colors(self):
        return map(int, self.colors[self.cur_marker]), self.cur_marker

    def watershed(self):
        m = self.markers.copy()
        cv2.watershed(self.img, m)
        self.truth=m
        overlay = self.colors[np.maximum(m, 0)]
        self.vis = cv2.addWeighted(self.img, 0.5, overlay, 0.5, 0.0, dtype=cv2.CV_8UC3)
        sub_vis=self.sketch.subimage.get_sub_image_of_this(self.vis)
        cv2.imshow('watershed', sub_vis)
    def show_all(self):
        self.sketch.show()
        self.watershed()
    def run(self):
        while True:
            ch = 0xFF & cv2.waitKey(50)
            if ch == 27:
                break
            if ch >= ord('1') and ch <= ord('7'):
                self.cur_marker = ch - ord('0')
                print('marker: ', self.cur_marker,self.colorlist[int(self.cur_marker)])
            if ch == ord(' ') or (self.sketch.dirty and self.auto_update):
                self.watershed()
                self.sketch.dirty = False
            if ch in [ord('a'), ord('A')]:
                self.auto_update = not self.auto_update
                print('auto_update if', ['off', 'on'][self.auto_update])
            if ch in [ord('r'), ord('R')]:
                self.markers[:] = 0
                self.watershed()
                self.markers_vis[:] = self.img
                self.sketch.subimage.reset_zoom()
                self.sketch.show()
            if ch in [ord('d'),ord('D')]:
                self.sketch.subimage.zoom_in()
                self.show_all()
            if ch in [ord('f'),ord('F')]:
                self.sketch.subimage.reset_subimage()
                self.show_all()           
        cv2.destroyAllWindows()


#if __name__ == '__main__':
#    import sys
#    pcfoldername="C:\Users\melocaladmin\My Documents\WormPics"
#    pcdataname="\\20140908 Confocal ZIM294 Step Size Stacks Split\\14-09-08 Confocal ZIM294 L4 W1 H 1.5um Stack Split\\14-09-08 Confocal ZIM294 L4 W1 H 1.5um Stack Split_7.tif"
#    path=pcfoldername+pcdataname
#    try: fn = sys.argv[1]
##    except: fn = '../cpp/fruits.jpg'
#    except: fn=path
#    print __doc__
#    WaterApp(fn).run()
