#example_tiled_sphere_compound.py

from spherical_tiling_3d.wythoff import WythoffPolyhedron
from spherical_tiling_3d.tiled_sphere import TiledSphere

poly = WythoffPolyhedron(5, 3, radius=50.0).build()
tiled = TiledSphere(poly, radius=50.0).build()

compound = tiled.to_compound()
show_object(compound)