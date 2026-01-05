#### wythoff.py

from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from scipy.spatial import ConvexHull

from .geometry import(
	reflect_point,
	group_triangles_by_plane,
	boundary_polygon_from_triangles
)


@dataclass
class PolyhedronFace:
	"""One polygonal face of the Wythoff polyhedron."""
	face_id: int
	boundary_vertex_indices: list[int]
	color: tuple[float, float, float] | None = None
	depth: float = 0.0
	meta: dict = field(default_factory=dict)


class WythoffPolyhedron:
	"""
	Regular {p, q} spherical polyhedron built via Wythoff construction
	"""

	def __init__(self, p: int, q: int, radius: float, rounding: int = 6):
		self.p = p
		self.q = q
		self.radius = radius
		self.rounding = rounding

		self.vertices: list[np.ndarray] = []
		self.faces: list[PolyhedronFace] = []
		self._face_adjacency: dict[unt, set[int]] | None = None


	def _compute_reflection_normals(self) -> list[np.ndarray]:
		"""
		Compute the three reflection normals for the {p, q} Wythoff construction
		"""
		p, q = self.p, self.q

		if 1 / p + 1 / q <= 0.5:
			raise ValueError("Invalid Schläfli symbol: 1/p + 1/q must be > 1/2 for spherical tilings.")

		n1 = np.array([1.0, 0.0, 0.0])
		n2 = np.array([0.0, 1.0, 0.0])

		cos_p = np.cos(np.pi / p)
		cos_q = np.cos(np.pi / q)

		z_sq = 1.0 - cos_p**2 - cos_q**2
		if z_sq < 0:
			if z_sq > -1e-10:
				z_sq = 0.0
			else:
				raise ValueError("Numerical error: z^2 is negative")
		z = np.sqrt(z_sq)
		n3 = np.array([-cos_q, -cos_p, z])

		return [n1, n2, n3]


	def _compute_orbit_seed(self, normals: list[np.ndarray]) -> np.ndarray:
		"""
		Compute the initial seed point for the Wythoff orbit.
		"""
		r = self.radius
		n1, _, n3 = normals

		seed_dir = np.cross(n1, n3)
		seed_dir /= np.linalg.norm(seed_dir)

		return seed_dir * r


	def _vertex_direction_key(self, p: np.ndarray) -> tuple[float, float, float]:
		"""
		Map a 3D point to a rounded unit direction key on the sphere.
		"""

		u = p / np.linalg.norm(p)
		return tuple(np.round(u, self.rounding))


	def _vertex_orbit_on_sphere(self) -> list[np.ndarray]:
		"""
		Compute the vertex orbit for the regular {p,q} case on a sphere of given radius.
		"""
		normals = self._compute_reflection_normals()
		seed = self._compute_orbit_seed(normals)

		visited: dict[tuple[float, float, float], np.ndarray] = {
			self._vertex_direction_key(seed): seed
		}
		frontier: list[np.ndarray] = [seed]

		while frontier:
			current = frontier.pop()
			for normal in normals:
				v = reflect_point(current, normal)
				v = v / np.linalg.norm(v) * self.radius
				k = self._vertex_direction_key(v)
				if k not in visited:
					visited[k] = v
					frontier.append(v)

		return list(visited.values())

	def get_face_adjacency(self) -> dict[int, set[int]]:
		"""
		Compute which faces share an edge
		"""
		if self._face_adjacency is not None:
			return self._face_adjacency

		# Map each edge to the face that contains it
		edge_to_faces: dict[tuple[int, int], list[int]] = defaultdict(list)

		for face in self.faces:
			indicies = face.boundary_vertex_indices
			n = len(indicies)

			for i in range(n):
				v1 = indicies[i]
				c2 = indicies[(i+1) % n]
				edge = (min(v1, v2), max(v1, v2))
				edge_to_faces[edge].append(face.face_id)

		# Build adjacency from shared edges
		adjacency: dict[int, set[int]] = defaultdict(set)

		for edge, face_ids in edge_to_faces.items():
			if len(face_ids) == 2:
				f1, f2 = face_ids
				adjacency[f1].add(f2)
				adjacency[f2].add(f1)
		self._face_adjacency = dict(adjacency)
		return self._face_adjacency


	def color_faces_greedy(self) -> dict[int, int]:
		"""
		Assign colors to faces such that not two adjacent faces share a color.
		Uses a greedy algorithm.
		"""
		adjacency = self.get_face_adjacency()
		coloring: dict[int, int] = {}

		for face in self.faces:
			face_id = face.face_id
			neighbor_colors = {
				coloring[neighbor_id]
				for neighbor_id in adjacency.get(face_id, set())
				if neighbor_id in coloring
			}

			color = 0
			while color in neighbor_colors:
				color += 1

			coloring[face_id] = color

		return coloring


	def build(self) -> "WythoffPolyhedron":
		"""
		Compute vertices and polygonal faces for the {p, q} polyhedron.
		"""

		self.vertices = self._vertex_orbit_on_sphere()

		P = np.asarray(self.vertices, float)
		hull = ConvexHull(P, qhull_options = "QJ")
		triangles = hull.simplices

		planes = group_triangles_by_plane(self.vertices, triangles, nd=self.rounding)

		faces: list[PolyhedronFace] = []
		for face_id, tri_list in enumerate(planes.values()):
			loop = boundary_polygon_from_triangles(tri_list)
			faces.append(
				PolyhedronFace(
					face_id=face_id,
					boundary_vertex_indices=loop
				)
			)

		self.faces = faces
		self._face_adjacency = None
		return self
