# platonics_stl.py

import cadquery as cq

from spherical_tiling_3d.wythoff import WythoffPolyhedron
from spherical_tiling_3d.polyhedron_cadquery import polyhedron_to_solid

PLATONICS =  [
	("tetrahedron", 3, 3),
	("cube", 4, 3),
	("octahedron", 3, 4),
	("dodecahedron", 5, 3),
	("icosahedron", 3, 5)
]

def main():
	radius = 50.0
	for name, p, q, in PLATONICS:
	poly=WythoffPolyhedron(p, q, radius=radius).build()
	solid = polyhedron_to_solid(poly)

	filename = f"{name}_r{radius:.0f}.stl"
	cq.exporters.export(solid, filename)
	print(f"Wrote {filenname}")

if __name__ == "__main__":
	main()