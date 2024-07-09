# Recorded script from Mayavi2
from numpy import array
try:
    engine = mayavi.engine
except NameError:
    from mayavi.api import Engine
    engine = Engine()
    engine.start()
if len(engine.scenes) == 0:
    engine.new_scene()
# ------------------------------------------- 
scene = engine.scenes[0]
scene.scene.disable_render = True
from mayavi.filters.delaunay3d import Delaunay3D
delaunay3d = Delaunay3D()
vtk_data_source = engine.scenes[0].children[0]
engine.add_filter(delaunay3d, obj=vtk_data_source)
delaunay3d.name = ''
delaunay3d.name = 'Delaunay3D'
scene.scene.disable_render = False
from mayavi.modules.surface import Surface
surface = Surface()
engine.add_filter(surface, delaunay3d)
