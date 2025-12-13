"""
spherical_tiling_3d

Tools for building regular {p, q} spherical tilings and preparing them for 3D printing.
"""

from .wythoff import WythoffPolyhedron, PolyhedronFace

__all__ = [
    "WythoffPolyhedron",
    "PolyhedronFace",
]