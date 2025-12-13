# polyhedron_cadquery.py

import cadquery as cq
import numpy as np

from. wythoff import WythoffPoly, PolyhedronFace

def _make_cq_face_from_polyhedron_face(
	vertices: list[np.ndarray],
	face: PolyhedronFace
) -> cq.Face:
	"""
	 Convert one PolyhedronFace into a CadQuery Face.
    - Takes the boundary vertex indices
    - Builds straight edges between them
    - Assembles a wire and then a planar face
	"""
	P = np.asarray(vertices, float)
	idx = list(face.boundary_vertex_indices)

	# ensure consistent orientation (outward normal)
	a, b, c = P[idx[0]], P[idx[1]], P[idx[2]]
	if np.dot(np.cross(b - a, c - a), a) < 0:
		idx.reverse()

	edges = []
	for i in range(len(idx)):
		p0 = cq.Vector(*P[idx[i]])
		p1 = cq.Vector(*P[idx[(i + 1) % len(idx)]])
		edges.append(cq.Edge.makeLine(p0, p1))

	wire = cq.Wire.assembleEdges(edges)
	return cq.Face.makeFromWires(wire)

def polyhedron_to_solid(poly: WythoffPolyhedron) -> cq.Solid:
	"""
	Build a closed CadQuery Solid from a Wythoff polyhedron.

	If the polyhedron is not built yet, this calls `build()` first.
	"""
	if not poly.vertices or not poly.faces:
		poly = poly.build()

	cq_faces = [
		_make_cq_face_from_polyhedron_face(poly.vertices, face)
		for face in poly.faces
	]

	shell = cq.Shell.makeShell(cq_faces)
	solid = cq.Solid.makeSolid(shell)
	return solid
