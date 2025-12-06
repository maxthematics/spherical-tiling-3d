#### geometry.py

import numpy as np
from collections import defaultdict, Counter

def reflect_point(point: np.ndarray, normal: np.ndarray) -> np.ndarray:
	"""Reflect a point across a plane through the origin with given normal."""
	n = normal / np.linalg.norm(normal)
	return point - 2 * np.dot(n, point) * n

def normalized_plane_key(
	A: np.ndarray,
	B: np.ndarray,
	C: np.ndarray,
	nd: int = 6,
) -> tuple[tuple[float, float, float], float] | None:
	"""
	Compute a normalized key representing the plane through three points
	"""

	A, B, C = map(lambda p: np.asarray(p, float), (A, B, C))
	n = np.cross(B - A, C - A)
	n_norm = np.linalg.norm(n)
	if n_norm < 1e-12:
		return None

	n /= n_norm
	if np.dot(n, A) < 0:
		n = -n
	d = float(np.dot(n, A))

	n_key = tuple(np.round(n, nd))
	d_key = round(d, nd)

	return n_key, d_key


def group_triangles_by_plane(
	points: list[np.ndarray] | np.ndarray,
	triangles: np.ndarray,
	nd: int = 6,
) -> dict[tuple[tuple[float, float, float], float], list[tuple[int, int, int]]]:
	"""
	Cluster triangles (given by vertex indices) by the plane they lie in.
	"""
	planes: dict[
				tuple[
					tuple[float, float, float],
					float
				],
				list[
					tuple[int, int, int]
				]
			] = defaultdict(list)
	P = points

	for (i, j, k) in triangles:
		key = normalized_plane_key(P[i], P[j], P[k], nd=nd)
		if key is not None:
			planes[key].append((int(i), int(j), int(k)))

	return planes


def boundary_polygon_from_triangles(
	tri_list: list[tuple[int, int, int]]
) -> list [int]:
	"""
	Given a set of coplanar triangles forming a convex patch, extract
	the boundary vertex cycle.
	"""
	edge_counter = Counter()
	for (i, j, k) in tri_list:
		for a, b in ((i, j), (j, k), (k, i)):
			e = (a,b) if a < b else (b, a)
			edge_counter[e] += 1

	boundary_edges = [e for e, c in edge_counter.items() if c == 1]

	adjacency: dict[int, list[int]] = defaultdict(list)
	for a, b in boundary_edges:
		adjacency[a].append(b)
		adjacency[b].append(a)

	start = boundary_edges[0][0]
	loop = [start]
	prev, curr = None, start

	while True:
		nbrs = adjacency[curr]
		if len(nbrs) == 1:
			nxt = nbrs[0]
		else:
			nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]
		if nxt == start:
			break
		loop.append(nxt)
		prev, curr = curr, nxt

	return loop





