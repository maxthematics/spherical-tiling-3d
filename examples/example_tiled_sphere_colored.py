# example_tiled_sphere_colored.py

from spherical_tiling_3d.wythoff import WythoffPolyhedron
from spherical_tiling_3d.tiled_sphere import TiledSphere

# Auswahl: "random", "greedy", "symmetric", "geometric"
COLOR_METHOD = "geometric"

# Platonic solids: (p, q)
# Tetrahedron: (3, 3), cube: (4, 3), octahedron: (3, 4)
# Dodecahedron: (5, 3), icosahedron: (3, 5)
p, q = 3, 5  # Ikosaeder

poly = WythoffPolyhedron(p, q, radius=50.0).build()
tiled = TiledSphere(poly, radius=50.0).build()

if COLOR_METHOD == "greedy":
    coloring = poly.color_faces_greedy()
    tiled.apply_coloring(coloring)
elif COLOR_METHOD == "symmetric":
    coloring = poly.color_faces_symmetric()
    tiled.apply_coloring(coloring)
elif COLOR_METHOD == "geometric":
    coloring = poly.color_faces_geometric()
    tiled.apply_coloring(coloring)
# with "random," the random colors from the constructor remain

for i, (solid, color) in enumerate(tiled.show_colored()):
    show_object(solid, name=f"tile_{i}", options={"color": color})