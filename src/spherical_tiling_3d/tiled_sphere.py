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

    DEFAULT_COLOR_PALETTE = [
        (1.0, 0.0, 0.0),   # red
        (0.0, 0.75, 0.0),  # green
        (0.0, 0.0, 1.0),   # blue
        (1.0, 1.0, 0.0),   # yellow

    ]
    
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

    
    def _make_tile(
        self, 
        face_index: int,
        hollow: bool = False,
        wall_thickness: float = 2.0
        ) -> cq.Solid:
        """
        Create a spherical tile by intersecting a pyramid with the sphere.
        """
        outer_sphere = cq.Solid.makeSphere(
            self.radius, angleDegrees1=-90, angleDegrees2=90
        )
        pyramid = self._make_pyramid_from_face(face_index)
        
        # Intersection via double cut: pyramid ∩ sphere = pyramid - (pyramid - sphere)
        tile_solid = pyramid.cut(pyramid.cut(outer_sphere))

        if hollow:
            inner_radius = self.radius - wall_thickness
            if inner_radius > 0:
                inner_sphere = cq.Solid.makeSphere(
                    inner_radius, angleDegrees1=-90, angleDegrees2=90
                )
                tile_solid = tile_solid.cut(inner_sphere)
        
        return tile_solid

    
    def build(
        self,
        hollow: bool = False,
        wall_thickness: float = 2.0
    ) -> "TiledSphere":
        """
        Build all spherical tiles from the polyhedron faces.
        """
        self.tiles = []
        
        for i in range(len(self.polyhedron.faces)):
            solid = self._make_tile(i, hollow=hollow, wall_thickness=wall_thickness)
            tile = SphericalTile(solid=solid, tile_id=i)
            self.tiles.append(tile)
        
        return self

        
    def apply_coloring(
        self,
        coloring: dict[int, int],
        palette: list[tuple[float, float, float]] | None = None
    ) -> "TiledSphere":
        """
        Apply a coloring to the tiles
        """

        if palette is None:
            palette = self.DEFAULT_COLOR_PALETTE

        for tile in self.tiles:
            color_index = coloring.get(tile.tile_id, 0)
            tile.color = palette[color_index % len(palette)]

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

    
    def group_by_color(self) -> dict[tuple[float, float, float], list[SphericalTile]]:
        """
        Group tiles by their color.
        
        Returns a dict mapping color tuples to lists of tiles.
        """
        groups: dict[tuple[float, float, float], list[SphericalTile]] = {}
        
        for tile in self.tiles:
            if tile.color not in groups:
                groups[tile.color] = []
            groups[tile.color].append(tile)
        
        return groups
    
    def export_stl_by_color(
        self, 
        output_dir: str, 
        base_name: str = "tile_color"
    ) -> list[str]:
        """
        Export tiles grouped by color as separate STL files.
        
        Each color group becomes one STL file containing all tiles of that color
        combined into a single compound.
        
        Args:
            output_dir: Directory to save STL files.
            base_name: Base name for output files.
        
        Returns:
            List of created file paths.
        """
        from pathlib import Path
        
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        color_groups = self.group_by_color()
        created_files = []
        
        # Map colors to readable names
        color_names = {
            (1.0, 0.0, 0.0): "red",
            (0.0, 0.75, 0.0): "green",
            (0.0, 0.0, 1.0): "blue",
            (1.0, 1.0, 0.0): "yellow",
            (1.0, 0.5, 0.0): "orange",
            (0.5, 0.0, 0.5): "purple",
        }
        
        for i, (color, tiles) in enumerate(color_groups.items()):
            # Get color name or use index
            color_name = color_names.get(color, f"color_{i}")
            
            # Combine all tiles of this color into one compound
            solids = [tile.solid for tile in tiles]
            compound = cq.Compound.makeCompound(solids)
            
            # Export
            filename = out_path / f"{base_name}_{color_name}.stl"
            cq.exporters.export(compound, str(filename))
            created_files.append(str(filename))
        
        return created_files




