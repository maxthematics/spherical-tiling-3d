#### __init__.py

""""
spherical_tiling_3d

tools for building regular {p,q} spherical tilings and preparing them for 3D-printing.
"""

from .wythoff import WythoffPolyhedron, SphericalFace
from .mesh import SphericalMesh

__all__ =  [
	"WythoffPolyhedron",
    "SphericalFace",
    "SphericalMesh",
]