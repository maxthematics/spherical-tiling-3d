# platonics_stl.py

from pathlib import Path
import cadquery as cq

from spherical_tiling_3d.wythoff import WythoffPolyhedron
from spherical_tiling_3d.polyhedron_cadquery import polyhedron_to_solid

PLATONICS =  [
	("tetrahedron", 3, 3),
	("cube", 4, 3),
	("octahedron", 3, 4),
	("dodecahedron", 5, 3),
	("icosahedron", 3, 5),
]

def main():
	radius = 50.0

	here = Path(__file__).resolve().parent
	out_dir = here / "stl"
	out_dir.mkdir(exist_ok=True)

	for name, p, q, in PLATONICS:
		poly=WythoffPolyhedron(p, q, radius=radius).build()
		solid = polyhedron_to_solid(poly)

		filename = out_dir / f"{name}_r{radius:.0f}.stl"
		cq.exporters.export(solid, str(filename))
		print(f"Wrote {filename}")

if __name__ == "__main__":
	main()