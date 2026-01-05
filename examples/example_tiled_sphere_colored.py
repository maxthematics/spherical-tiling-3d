#example_tiled_sphere_colored.py

from spherical_tiling_3d.wythoff import WythoffPolyhedron
from spherical_tiling_3d.tiled_sphere import TiledSphere

# Ikosaeder-basierte Kugel
poly = WythoffPolyhedron(3, 5, radius=50.0).build()
tiled = TiledSphere(poly, radius=50.0).build()

for solid, color in tiled.show_colored():
    show_object(solid, options={"color": color})