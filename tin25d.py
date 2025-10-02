from __future__ import annotations
import sys, os, math
import numpy as np


# Check if the first line of a CSV file is a header (contains letters)
def _has_header(path: str) -> bool:
	with open(path, "r", encoding="utf-8", errors="ignore") as f:
		first = f.readline()
	# Heuristic: any alpha in first line => header
	return any(ch.isalpha() for ch in first)


# Load x, y, z columns from a CSV file, skipping header if present
def _load_xyz(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	has_hdr = _has_header(path)
	data = np.loadtxt(path, delimiter=",", skiprows=1 if has_hdr else 0)
	if data.ndim == 1:
		data = data[None, :]
	if data.shape[1] < 3:
		raise ValueError("CSV must have at least 3 numeric columns: x,y,z")
	x, y, z = data[:, 0], data[:, 1], data[:, 2]
	return x, y, z


# Remove duplicate or nearly identical XY points by rounding and averaging their coordinates
def _dedupe_xy(x: np.ndarray, y: np.ndarray, z: np.ndarray, tol_decimals: int = 8):
	# Group near-identical XY by rounding, then average x,y,z per group
	xy_round = np.round(np.c_[x, y], tol_decimals)
	_, inv = np.unique(xy_round, axis=0, return_inverse=True)
	counts = np.bincount(inv)
	x_u = np.bincount(inv, weights=x) / counts
	y_u = np.bincount(inv, weights=y) / counts
	z_u = np.bincount(inv, weights=z) / counts
	return x_u, y_u, z_u


# Build a face adjacency matrix (each triangle’s neighbors across its three edges)
def _face_adjacency(F: np.ndarray) -> np.ndarray:
	if F.size == 0:
		return np.empty((0, 3), dtype=int)
	A = np.full((F.shape[0], 3), -1, dtype=int)
	edge_map: dict[tuple[int, int], tuple[int, int]] = {}
	for fi, (a, b, c) in enumerate(F):
		for ei, (u, v) in enumerate(((a, b), (b, c), (c, a))):
			key = (u, v) if u < v else (v, u)
			if key in edge_map:
				fj, ej = edge_map.pop(key)
				A[fi, ei] = fj
				A[fj, ej] = fi
			else:
				edge_map[key] = (fi, ei)
	return A


# Build a 2.5D TIN: load points, dedupe, triangulate in XY, attach Z, compute adjacency
def build_tin(csv_path: str):
	x, y, z = _load_xyz(csv_path)
	x, y, z = _dedupe_xy(x, y, z)
	pts2 = np.c_[x, y]
	if len(pts2) < 3:
		V = np.c_[x, y, z]
		F = np.empty((0, 3), dtype=int)
		A = np.empty((0, 3), dtype=int)
		return V, F, A
	try:
		from scipy.spatial import Delaunay  # type: ignore
	except Exception as e:
		raise ImportError(
			"SciPy is required for Delaunay triangulation. Install with: pip install scipy"
		) from e
	tri = Delaunay(pts2)
	F = tri.simplices.astype(int)
	V = np.c_[x, y, z]
	A = _face_adjacency(F)
	return V, F, A
