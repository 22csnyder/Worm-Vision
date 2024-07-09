# -*- coding: utf-8 -*-
"""
Created on Mon May 11 05:30:15 2015

@author: cgs567
"""
import math

def bresenham_march(img, p1, p2):
   x1 = p1[0]
   y1 = p1[1]
   x2 = p2[0]
   y2 = p2[1]
   steep = math.fabs(y2 - y1) > math.fabs(x2 - x1)
   if steep:
      t = x1
      x1 = y1
      y1 = t

      t = x2
      x2 = y2
      y2 = t
   also_steep = x1 > x2
   if also_steep:

      t = x1
      x1 = x2
      x2 = t

      t = y1
      y1 = y2
      y2 = t

   dx = x2 - x1
   dy = math.fabs(y2 - y1)
   error = 0.0
   delta_error = 0.0; # Default if dx is zero
   if dx != 0:
       delta_error = math.fabs(dy/dx)

   if y1 < y2:
      y_step = 1 
   else:
      y_step = -1

   y = y1
   ret = list([])
   for x in range(x1, x2):
      if steep:
         p = (y, x)
      else:
         p = (x, y)

      (b,r,g,a) = (-1,)*4
      if p[0] < img.width and p[1] < img.height:
         (b,r,g, a) = cv.Get2D(img, p[1], p[0])

      ret.append((p,(b,r,g)))

      error += delta_error
      if error >= 0.5:
         y += y_step
         error -= 1

   if also_steep:
       ret.reverse()

   return ret