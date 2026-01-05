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
				v2 = indicies[(i+1) % n]
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



	def _get_face_centroids(self) -> dict[int, np.ndarray]:
		"""Calculate the centroid for each face."""
		centroids = {}
		for face in self.faces:
			coords = [self.vertices[i] for i in face.boundary_vertex_indices]
			centroids[face.face_id] = np.mean(coords, axis=0)
		return centroids

	def _group_antipodal_faces(self) -> list[list[int]]:
		"""
		Group faces that are opposite (antipodal) to each other.
		
		Antipodal faces have centroids pointing in opposite directions.
		These pairs can share a color for inversion symmetry.
		"""
		centroids = self._get_face_centroids()
		face_ids = set(f.face_id for f in self.faces)
		groups = []
		visited = set()

		for fid in face_ids:
			if fid in visited:
				continue
			
			center = centroids[fid]
			match_found = None
			
			for candidate_id in face_ids:
				if candidate_id == fid or candidate_id in visited:
					continue
				
				# Check if centroids are antipodal (c + other ≈ 0)
				if np.linalg.norm(center + centroids[candidate_id]) < 1e-4 * self.radius:
					match_found = candidate_id
					break
			
			if match_found is not None:
				groups.append([fid, match_found])
				visited.add(fid)
				visited.add(match_found)
			else:
				groups.append([fid])
				visited.add(fid)
				
		return groups


	def _recursive_coloring_solver(
		self,
		group_index: int,
		groups: list[list[int]], 
		adjacency: dict[int, set[int]],
		coloring: dict[int, int], 
		max_colors: int
	) -> bool:
		"""Backtracking solver for graph coloring on groups of faces."""
		if group_index == len(groups):
			return True

		current_group = groups[group_index]
		
		for color in range(max_colors):
			is_safe = True
			for face_id in current_group:
				neighbors = adjacency.get(face_id, set())
				for n_id in neighbors:
					if n_id in coloring and coloring[n_id] == color:
						is_safe = False
						break
				if not is_safe:
					break
			
			if is_safe:
				for face_id in current_group:
					coloring[face_id] = color
				
				if self._recursive_coloring_solver(
					group_index + 1, groups, adjacency, coloring, max_colors
				):
					return True
				
				for face_id in current_group:
					del coloring[face_id]

		return False


	def color_faces_symmetric(self, max_colors: int = 4) -> dict[int, int]:
		"""
		Color faces with inversion symmetry (antipodal faces share color).
		
		Falls back to standard coloring if symmetric coloring is impossible.
		
		Returns a dict mapping face_id to color_index (0, 1, 2, ...).
		"""
		adjacency = self.get_face_adjacency()
		
		# Try strict symmetry (antipodal pairs share color)
		symmetric_groups = self._group_antipodal_faces()
		coloring: dict[int, int] = {}
		
		success = self._recursive_coloring_solver(
			0, symmetric_groups, adjacency, coloring, max_colors
		)
		
		if success:
			return coloring
		
		# Fallback: solve without symmetry constraint
		single_groups = [[f.face_id] for f in self.faces]
		coloring = {}
		
		# Heuristic: most constrained first
		single_groups.sort(
			key=lambda g: len(adjacency.get(g[0], set())), 
			reverse=True
		)

		success = self._recursive_coloring_solver(
			0, single_groups, adjacency, coloring, max_colors
		)
		
		if success:
			return coloring
		else:
			raise ValueError(
				f"Could not color polyhedron with {max_colors} colors"
			)


	def _get_geometric_independent_sets(
		self, 
		size_per_color: int, 
		tolerance: float = 0.05
	) -> list[set[int]]:
		"""
		Find all face sets that are independent and geometrically regular.
		
		A set is valid if:
		1. No two faces share an edge (independent)
		2. All pairwise centroid distances are equal (regular)
		"""
		centroids = self._get_face_centroids()
		adjacency = self.get_face_adjacency()
		face_ids = list(f.face_id for f in self.faces)
		
		valid_sets = []

		def find_sets_recursive(current_set: list[int], start_index: int):
			# Pruning: check independence immediately
			if len(current_set) > 1:
				last = current_set[-1]
				neighbors = adjacency.get(last, set())
				for other in current_set[:-1]:
					if other in neighbors:
						return

			if len(current_set) == size_per_color:
				# Check geometric regularity
				if size_per_color == 1:
					valid_sets.append(set(current_set))
					return
					
				dists = []
				c_vecs = [centroids[fid] for fid in current_set]
				for i in range(len(c_vecs)):
					for j in range(i + 1, len(c_vecs)):
						dists.append(np.linalg.norm(c_vecs[i] - c_vecs[j]))
				
				# All distances should be nearly equal
				if np.std(dists) < (np.mean(dists) * tolerance):
					valid_sets.append(set(current_set))
				return

			for i in range(start_index, len(face_ids)):
				find_sets_recursive(current_set + [face_ids[i]], i + 1)

		find_sets_recursive([], 0)
		
		return valid_sets


	def _solve_exact_cover(
		self, 
		candidate_sets: list[set[int]], 
		target_colors: int
	) -> dict[int, int] | None:
		"""
		Select target_colors disjoint sets that cover all faces (Exact Cover).
		"""
		all_faces = set(f.face_id for f in self.faces)
		
		def backtrack(
			current_selection: list[set[int]], 
			covered_faces: set[int],
			start_index: int
		):
			if len(current_selection) == target_colors:
				if covered_faces == all_faces:
					return current_selection
				return None
			
			for i in range(start_index, len(candidate_sets)):
				candidate = candidate_sets[i]
				if not candidate.isdisjoint(covered_faces):
					continue
				
				result = backtrack(
					current_selection + [candidate], 
					covered_faces | candidate,
					i + 1
				)
				if result:
					return result
			return None

		solution_sets = backtrack([], set(), 0)
		
		if solution_sets:
			coloring = {}
			for color_idx, face_set in enumerate(solution_sets):
				for face_id in face_set:
					coloring[face_id] = color_idx
			return coloring
		return None


	def color_faces_geometric(self, num_colors: int | None = None) -> dict[int, int]:
		"""
		Color faces with maximum geometric symmetry.
		
		Finds groups of faces that form regular patterns (equal centroid distances)
		and assigns each group a color. This produces the classical symmetric
		colorings (e.g., 5-coloring of icosahedron based on compound of 5 tetrahedra).
		
		Args:
			num_colors: Number of colors to use. If None, automatically determined.
		
		Returns a dict mapping face_id to color_index (0, 1, 2, ...).
		"""
		num_faces = len(self.faces)
		
		# Auto-detect optimal number of colors for platonic solids
		if num_colors is None:
			if self.p == 3 and self.q == 3:    # Tetrahedron
				num_colors = 4
			elif self.p == 4 and self.q == 3:  # Cube
				num_colors = 3
			elif self.p == 3 and self.q == 4:  # Octahedron
				num_colors = 2
			elif self.p == 5 and self.q == 3:  # Dodecahedron
				num_colors = 4
			elif self.p == 3 and self.q == 5:  # Icosahedron
				num_colors = 5
			else:
				num_colors = 4  # Default fallback

		if num_faces % num_colors != 0:
			# Can't divide evenly, fall back to symmetric
			return self.color_faces_symmetric(num_colors)

		size_per_color = num_faces // num_colors
		
		# Find all geometrically regular independent sets
		candidates = self._get_geometric_independent_sets(size_per_color)
		
		if not candidates:
			return self.color_faces_symmetric(num_colors)

		# Solve exact cover problem
		coloring = self._solve_exact_cover(candidates, num_colors)
		
		if coloring:
			return coloring
		else:
			return self.color_faces_symmetric(num_colors)


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
