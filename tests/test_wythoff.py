import numpy as np
from spherical_tiling_3d.wythoff import WythoffPolyhedron


def test_icosahedron_3_5():
    poly = WythoffPolyhedron(3, 5, radius=1.0).build()

    # Icosahedron: 12 vertices, 20 faces
    assert len(poly.vertices) == 12
    assert len(poly.faces) == 20

   # All vertices should be on (almost) the same radius.
    radii = [np.linalg.norm(v) for v in poly.vertices]
    assert max(radii) - min(radii) < 1e-6