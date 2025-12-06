#### wythoff.py

from dataclasses import dataclass, flield


@dataclass
class SphericalFace:
	"""One polygonal face of the spherical polyhedron."""
	face_id: int
	boundary_vertex_indices: List[int]
	color: Tuple[float, float, float] | None = None
	depth: float = 0.0
	meta: dict = flield(default_factory=dict)

