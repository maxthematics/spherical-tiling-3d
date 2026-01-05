#tiled_sphere.py

from dataclasses import dataclass, field
import random

import cadquery as cq


def _random_color() -> tuple[float, float, float]:
    """Generate a random RGB color."""
    return (random.random(), random.random(), random.random())


@dataclass
class SphericalTile:
    """A single tile of a tiled sphere."""   
    solid: cq.Solid
    tile_id: int
    color: tuple[float, float, float] = field(default_factory=_random_color)


class TiledSphere:
    """
    A sphere tiled by projecting polyhedron faces onto its surface.
    
    Works with any convex polyhedron whose vertices lie on a circumsphere.
    """
    
    def __init__(self, polyhedron, radius: float, pyramid_scale: float = 4.0):
        """
        Initialize a tiled sphere.
        """
        self.polyhedron = polyhedron
        self.radius = radius
        self.pyramid_scale = pyramid_scale
        self.tiles: list[SphericalTile] = []
    
    def _make_pyramid_from_face(self, face_index: int) -> cq.Solid:
        """
        Create a pyramid from origin to a scaled face of the polyhedron.
        """
        face = self.polyhedron.faces[face_index]
        apex = cq.Vector(0, 0, 0)
        
        # Collect and scale base points
        base_points = []
        for idx in face.boundary_vertex_indices:
            v = self.polyhedron.vertices[idx]
            scaled = v * self.pyramid_scale
            base_points.append(cq.Vector(*tuple(scaled)))
        
        n = len(base_points)
        faces = []
        
        # Side faces (triangles from apex to base edges)
        for i in range(n):
            p1 = base_points[i]
            p2 = base_points[(i + 1) % n]
            triangle_wire = cq.Wire.makePolygon([apex, p1, p2, apex])
            triangle_face = cq.Face.makeFromWires(triangle_wire)
            faces.append(triangle_face)
        
        # Base face
        base_wire = cq.Wire.makePolygon(base_points + [base_points[0]])
        base_face = cq.Face.makeFromWires(base_wire)
        faces.append(base_face)
        
        shell = cq.Shell.makeShell(faces)
        solid = cq.Solid.makeSolid(shell)
        
        return solid

    
    def _make_tile(self, face_index: int) -> cq.Solid:
        """
        Create a spherical tile by intersecting a pyramid with the sphere.
        """
        sphere = cq.Solid.makeSphere(self.radius, angleDegrees1=-90, angleDegrees2=90)
        pyramid = self._make_pyramid_from_face(face_index)
        
        # Intersection via double cut: pyramid ∩ sphere = pyramid - (pyramid - sphere)
        tile_solid = pyramid.cut(pyramid.cut(sphere))
        
        return tile_solid

    
    def build(self) -> "TiledSphere":
        """
        Build all spherical tiles from the polyhedron faces.
        """
        self.tiles = []
        
        for i in range(len(self.polyhedron.faces)):
            solid = self._make_tile(i)
            tile = SphericalTile(solid=solid, tile_id=i)
            self.tiles.append(tile)
        
        return self

    
    def to_compound(self) -> cq.Compound:
        """
        Combine all tiles into a single compound (without color information).
        """
        solids = [tile.solid for tile in self.tiles]
        return cq.Compound.makeCompound(solids)
        
    
    def show_colored(self) -> list[tuple[cq.Solid, tuple[float, float, float]]]:
        """
        Return a list of (solid, color) tuples for colored visualization.
        
        Usage in CQ-Editor:
            for solid, color in tiled_sphere.show_colored():
                show_object(solid, options={"color": color})
        """
        return [(tile.solid, tile.color) for tile in self.tiles]










