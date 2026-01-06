#export_colored_platonics.py

from pathlib import Path
from spherical_tiling_3d.wythoff import WythoffPolyhedron
from spherical_tiling_3d.tiled_sphere import TiledSphere

PLATONICS = [
    ("tetrahedron", 3, 3),
    ("cube", 4, 3),
    ("octahedron", 3, 4),
    ("dodecahedron", 5, 3),
    ("icosahedron", 3, 5),
]

def main():
    radius = 50.0
    
    # Output-Verzeichnis relativ zum Skript
    here = Path(__file__).resolve().parent
    out_dir = here / "stl"
    out_dir.mkdir(exist_ok=True)
    
    for name, p, q in PLATONICS:
        print(f"\n{'='*40}")
        print(f"Generating {name} {{p={p}, q={q}}}...")
        
        # Polyeder und Tiled Sphere bauen
        poly = WythoffPolyhedron(p, q, radius=radius).build()
        tiled = TiledSphere(poly, radius=radius).build(hollow=True, wall_thickness=3.0)
        
        # Geometrisch-symmetrische Färbung anwenden
        coloring = poly.color_faces_symmetric()
        tiled.apply_coloring(coloring)
        
        # STLs nach Farbe exportieren
        files = tiled.export_stl_by_color(str(out_dir), base_name=name)
        
        for f in files:
            print(f"  Wrote {f}")
    
    print(f"\n{'='*40}")
    print(f"Done! All files saved to {out_dir}")

if __name__ == "__main__":
    main()