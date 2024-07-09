# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
It's mostly scratch code from examples
"""






#to move from filters to console: do instead
pdi=servermanager.Fetch(FindSource("Sphere1"))

# This filter computes the volume of the tetrahedra in an unstructured mesh:
pdi = self.GetInput()
pdo = self.GetOutput()
newData = vtk.vtkDoubleArray()
newData.SetName("Volume")
numTets = pdi.GetNumberOfCells()
for i in range(numTets):
       cell = pdi.GetCell(i)
       p1 = pdi.GetPoint(cell.GetPointId(0))
       p2 = pdi.GetPoint(cell.GetPointId(1))
       p3 = pdi.GetPoint(cell.GetPointId(2))
       p4 = pdi.GetPoint(cell.GetPointId(3))
       volume = vtk.vtkTetra.ComputeVolume(p1,p2,p3,p4)
       newData.InsertNextValue(volume)
pdo.GetCellData().AddArray(newData)


##show a sphere and its coordinates in a spreadsheet 

try: paraview.simple
except: from paraview.simple import *
paraview.simple._DisableFirstRenderCameraReset()

Sphere1 = Sphere()

RenderView1 = GetRenderView()

DataRepresentation1 = Show()
DataRepresentation1.ScaleFactor = 0.1
DataRepresentation1.SelectionPointFieldDataArrayName = 'Normals'
DataRepresentation1.EdgeColor = [0.0, 0.0, 0.5000076295109483]

RenderView1.CameraViewUp = [0.5783107779384218, 0.26121825390557263, -0.7728658796626886]
RenderView1.CameraPosition = [-1.6838787253005265, -2.0453709888861047, -1.9513003401659341]
RenderView1.CameraClippingRange = [1.5679622628277057, 5.46647277653575]
RenderView1.CameraParallelScale = 0.8516115354228021

DataRepresentation1.Representation = 'Surface With Edges'

SpreadSheetView1 = CreateView( "SpreadSheetView" )

AnimationScene1 = GetAnimationScene()
AnimationScene1.ViewModules = [ RenderView1, SpreadSheetView1 ]

DataRepresentation2 = Show()

Render()