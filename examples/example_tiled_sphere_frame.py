# example_tiled_sphere_frame.py

from spherical_tiling_3d.wythoff import WythoffPolyhedron
from spherical_tiling_3d.tiled_sphere import TiledSphere

# Platonic solids: (p, q)
# Tetrahedron: (3, 3), cube: (4, 3), octahedron: (3, 4)
# Dodecahedron: (5, 3), icosahedron: (3, 5)
p, q = 3, 5  # Icosahedron

radius = 50.0
wall_thickness = 2.0
frame_width = 5.0

poly = WythoffPolyhedron(p, q, radius=radius).build()
tiled = TiledSphere(poly, radius=radius).build_frame(
    wall_thickness=wall_thickness,
    frame_width=frame_width
)

show_object(tiled.to_compound())