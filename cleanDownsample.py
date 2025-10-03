#!/usr/bin/env python3
"""cleanDownsample.py

Utility to:
 1. Load point cloud from CSV, PLY, PCD (ASCII), XYZ (simple whitespace) or OBJ (vertices only)
 2. Remove outliers via simple Z clipping (min/max) and/or statistical filter (mean +/- k*std on Z or full 3D radius based)
 3. Downsample using an XY voxel grid (aggregating Z by mean, keeping average color if available)
 4. Save cleaned cloud to CSV or PLY.

Design goals:
 - Zero heavy dependencies by default (only numpy & optional open3d if installed for PLY/PCD parsing)
 - Fallback pure-python parsers for simple formats
 - Deterministic output ordering (voxel cell sort) for reproducibility

Examples:
  python cleanDownsample.py -i "Point Cloud/point_cloud.csv" -o cleaned.csv --z-min 0.3 --z-max 2.0 -v 0.5
  python cleanDownsample.py -i input.ply -o cleaned.ply -v 1.0 --stat-z 2.5
  python cleanDownsample.py -i raw.xyz -o cleaned.csv -v 0.75 --stat-3d 2.0 --max-n 5_000_000

"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:  # optional dependency
	import open3d as o3d  # type: ignore
	_HAS_O3D = True
except Exception:  # pragma: no cover
	_HAS_O3D = False


@dataclass
class PointCloud:
	xyz: np.ndarray  # shape (N,3)
	rgb: Optional[np.ndarray] = None  # shape (N,3) uint8

	def ensure_float32(self):
		if self.xyz.dtype != np.float32:
			self.xyz = self.xyz.astype(np.float32)
		if self.rgb is not None and self.rgb.dtype != np.uint8:
			self.rgb = self.rgb.astype(np.uint8)


# --------------------------- Loading ---------------------------------- #
def load_point_cloud(path: str) -> PointCloud:
	ext = os.path.splitext(path)[1].lower()
	if ext == '.csv':
		return _load_csv(path)
	if ext == '.xyz':
		return _load_xyz(path)
	if ext == '.obj':
		return _load_obj(path)
	if ext == '.ply':
		return _load_ply(path)
	if ext == '.pcd':
		return _load_pcd(path)
	raise ValueError(f"Unsupported file extension: {ext}")


def _load_csv(path: str) -> PointCloud:
	# Expect header with x,y,z optionally r,g,b
	with open(path, 'r', newline='') as f:
		reader = csv.reader(f)
		header = next(reader)
		header_lower = [h.strip().lower() for h in header]
		# identify column indices
		try:
			ix = header_lower.index('x')
			iy = header_lower.index('y')
			iz = header_lower.index('z')
		except ValueError:
			raise ValueError('CSV must contain x,y,z header')
		color_cols = []
		for c in ['r', 'g', 'b']:
			if c in header_lower:
				color_cols.append(header_lower.index(c))
		pts = []
		cols = []
		for row in reader:
			if not row:
				continue
			try:
				x = float(row[ix]); y = float(row[iy]); z = float(row[iz])
				pts.append((x, y, z))
				if len(color_cols) == 3:
					cols.append(tuple(int(float(row[i])) for i in color_cols))
			except Exception:
				continue
		xyz = np.asarray(pts, dtype=np.float32)
		rgb = np.asarray(cols, dtype=np.uint8) if cols else None
		return PointCloud(xyz, rgb)


def _load_xyz(path: str) -> PointCloud:
	pts = []
	with open(path, 'r') as f:
		for line in f:
			line = line.strip()
			if not line or line.startswith('#'):
				continue
			parts = line.replace(',', ' ').split()
			if len(parts) < 3:
				continue
			try:
				pts.append(tuple(float(p) for p in parts[:3]))
			except ValueError:
				continue
	return PointCloud(np.asarray(pts, dtype=np.float32))


def _load_obj(path: str) -> PointCloud:
	pts = []
	with open(path, 'r') as f:
		for line in f:
			if line.startswith('v '):
				parts = line.strip().split()
				if len(parts) >= 4:
					try:
						pts.append(tuple(float(p) for p in parts[1:4]))
					except ValueError:
						pass
	return PointCloud(np.asarray(pts, dtype=np.float32))


def _load_ply(path: str) -> PointCloud:
	if _HAS_O3D:
		mesh = o3d.io.read_point_cloud(path)
		xyz = np.asarray(mesh.points, dtype=np.float32)
		rgb = None
		if mesh.has_colors():
			rgb = np.clip(np.asarray(mesh.colors) * 255.0 + 0.5, 0, 255).astype(np.uint8)
		return PointCloud(xyz, rgb)
	# Minimal ASCII PLY fallback
	with open(path, 'r') as f:
		if f.readline().strip() != 'ply':
			raise ValueError('Not a PLY file')
		line = ''
		prop_names = []
		vertex_count = 0
		while True:
			line = f.readline()
			if not line:
				raise ValueError('Unexpected EOF in header')
			line = line.strip()
			if line.startswith('element vertex'):
				vertex_count = int(line.split()[-1])
			elif line.startswith('property'):
				prop_names.append(line.split()[-1])
			elif line == 'end_header':
				break
		pts = []
		cols = []
		for _ in range(vertex_count):
			parts = f.readline().split()
			if len(parts) < 3:
				continue
			x, y, z = map(float, parts[:3])
			pts.append((x, y, z))
			# attempt color
			if len(parts) >= 6 and {'red','green','blue'} <= set(prop_names):
				try:
					r, g, b = map(int, parts[3:6])
					cols.append((r, g, b))
				except ValueError:
					pass
		xyz = np.asarray(pts, dtype=np.float32)
		rgb = np.asarray(cols, dtype=np.uint8) if cols else None
		return PointCloud(xyz, rgb)


def _load_pcd(path: str) -> PointCloud:
	if _HAS_O3D:
		pc = o3d.io.read_point_cloud(path)
		xyz = np.asarray(pc.points, dtype=np.float32)
		rgb = None
		if pc.has_colors():
			rgb = np.clip(np.asarray(pc.colors) * 255.0 + 0.5, 0, 255).astype(np.uint8)
		return PointCloud(xyz, rgb)
	# Minimal ASCII parser (XYZRGB or XYZ)
	with open(path, 'r') as f:
		fields = []
		count = 0
		data_ascii = False
		for line in f:
			line = line.strip()
			if line.startswith('FIELDS'):
				fields = line.split()[1:]
			elif line.startswith('POINTS'):
				count = int(line.split()[1])
			elif line.startswith('DATA'):
				if 'ascii' not in line:
					raise ValueError('Only ASCII PCD supported without open3d')
				data_ascii = True
				break
		if not data_ascii:
			raise ValueError('PCD DATA section not found')
		pts = []
		cols = []
		for _ in range(count):
			parts = f.readline().split()
			if len(parts) < 3:
				continue
			x, y, z = map(float, parts[:3])
			pts.append((x, y, z))
			if len(parts) >= 6 and {'r','g','b'} <= set(fields):
				try:
					r, g, b = map(int, parts[3:6])
					cols.append((r,g,b))
				except ValueError:
					pass
		return PointCloud(np.asarray(pts, dtype=np.float32), np.asarray(cols, dtype=np.uint8) if cols else None)


# --------------------------- Filtering -------------------------------- #
def z_clip(pc: PointCloud, z_min: Optional[float], z_max: Optional[float]) -> PointCloud:
	if z_min is None and z_max is None:
		return pc
	z = pc.xyz[:, 2]
	mask = np.ones(len(z), dtype=bool)
	if z_min is not None:
		mask &= z >= z_min
	if z_max is not None:
		mask &= z <= z_max
	xyz = pc.xyz[mask]
	rgb = pc.rgb[mask] if pc.rgb is not None else None
	return PointCloud(xyz, rgb)


def statistical_z(pc: PointCloud, k_std: float) -> PointCloud:
	if k_std <= 0:
		return pc
	z = pc.xyz[:, 2]
	mu = float(z.mean())
	sigma = float(z.std()) or 1e-9
	mask = (z >= mu - k_std * sigma) & (z <= mu + k_std * sigma)
	xyz = pc.xyz[mask]
	rgb = pc.rgb[mask] if pc.rgb is not None else None
	return PointCloud(xyz, rgb)


def statistical_radius(pc: PointCloud, k_std: float, sample: int = 100000) -> PointCloud:
	"""Simple global 3D distance magnitude filter (not kNN) to remove far sparse outliers.
	Uses distance from centroid; retains points within mean +/- k*std of radial distance.
	"""
	if k_std <= 0:
		return pc
	xyz = pc.xyz
	if len(xyz) == 0:
		return pc
	if len(xyz) > sample:
		idx = np.random.default_rng(123).choice(len(xyz), sample, replace=False)
		subset = xyz[idx]
	else:
		subset = xyz
	centroid = subset.mean(axis=0)
	dists = np.linalg.norm(subset - centroid, axis=1)
	mu = float(dists.mean())
	sigma = float(dists.std()) or 1e-9
	full_dists = np.linalg.norm(xyz - centroid, axis=1)
	mask = (full_dists >= mu - k_std * sigma) & (full_dists <= mu + k_std * sigma)
	rgb = pc.rgb[mask] if pc.rgb is not None else None
	return PointCloud(xyz[mask], rgb)


# --------------------------- Downsample ------------------------------- #
def voxel_downsample_xy(pc: PointCloud, voxel: float) -> PointCloud:
	if voxel <= 0:
		return pc
	xyz = pc.xyz
	# compute voxel indices in XY only
	ix = np.floor(xyz[:, 0] / voxel).astype(np.int64)
	iy = np.floor(xyz[:, 1] / voxel).astype(np.int64)
	# key as combined 64-bit (assuming ranges moderate); alternative tuple approach slower
	key = (ix << 32) ^ (iy & 0xffffffff)
	order = np.argsort(key)
	key_sorted = key[order]
	xyz_sorted = xyz[order]
	rgb_sorted = pc.rgb[order] if pc.rgb is not None else None
	unique_keys, start_idx = np.unique(key_sorted, return_index=True)
	# add sentinel for easier iteration
	start_idx = np.append(start_idx, len(key_sorted))
	out_pts = []
	out_cols = [] if rgb_sorted is not None else None
	for i in range(len(unique_keys)):
		s = start_idx[i]; e = start_idx[i+1]
		chunk = xyz_sorted[s:e]
		mean_xyz = chunk.mean(axis=0)
		out_pts.append(mean_xyz)
		if out_cols is not None:
			mean_rgb = rgb_sorted[s:e].mean(axis=0)
			out_cols.append(mean_rgb)
	new_xyz = np.vstack(out_pts).astype(np.float32)
	new_rgb = None
	if out_cols is not None:
		new_rgb = np.vstack(out_cols).clip(0,255).astype(np.uint8)
	# Sort by XY for deterministic output
	order2 = np.lexsort((new_xyz[:,1], new_xyz[:,0]))
	new_xyz = new_xyz[order2]
	if new_rgb is not None:
		new_rgb = new_rgb[order2]
	return PointCloud(new_xyz, new_rgb)


# --------------------------- Saving ---------------------------------- #
def save_point_cloud(pc: PointCloud, path: str):
	ext = os.path.splitext(path)[1].lower()
	if ext == '.csv':
		_save_csv(pc, path)
	elif ext == '.ply':
		_save_ply(pc, path)
	else:
		# default to csv
		_save_csv(pc, path)


def _save_csv(pc: PointCloud, path: str):
	with open(path, 'w', newline='') as f:
		w = csv.writer(f)
		if pc.rgb is not None:
			w.writerow(['x','y','z','r','g','b'])
			for (x,y,z), (r,g,b) in zip(pc.xyz, pc.rgb):
				w.writerow([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", int(r), int(g), int(b)])
		else:
			w.writerow(['x','y','z'])
			for x,y,z in pc.xyz:
				w.writerow([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])


def _save_ply(pc: PointCloud, path: str):
	if _HAS_O3D:
		p = o3d.geometry.PointCloud()
		p.points = o3d.utility.Vector3dVector(pc.xyz.astype(np.float64))
		if pc.rgb is not None:
			p.colors = o3d.utility.Vector3dVector((pc.rgb / 255.0).astype(np.float64))
		o3d.io.write_point_cloud(path, p, write_ascii=True)
		return
	# minimal ASCII PLY
	has_color = pc.rgb is not None
	with open(path, 'w') as f:
		f.write('ply\nformat ascii 1.0\n')
		f.write(f'element vertex {len(pc.xyz)}\n')
		f.write('property float x\nproperty float y\nproperty float z\n')
		if has_color:
			f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
		f.write('end_header\n')
		if has_color:
			for (x,y,z),(r,g,b) in zip(pc.xyz, pc.rgb):
				f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
		else:
			for (x,y,z) in pc.xyz:
				f.write(f"{x} {y} {z}\n")


# --------------------------- CLI ------------------------------------- #
def parse_args(argv=None):
	p = argparse.ArgumentParser(description='Clean and downsample a point cloud')
	p.add_argument('-i','--input', required=True, help='Input point cloud file (csv, ply, xyz, pcd, obj)')
	p.add_argument('-o','--output', required=True, help='Output file (csv or ply)')
	p.add_argument('-v','--voxel', type=float, default=0.0, help='XY voxel size (meters). 0 disables downsampling.')
	p.add_argument('--target-n', type=int, default=None, help='If set, ignore --voxel and adaptively choose voxel size to approach this point count (XY only).')
	p.add_argument('--z-min', type=float, default=None, help='Z minimum clip')
	p.add_argument('--z-max', type=float, default=None, help='Z maximum clip')
	p.add_argument('--stat-z', type=float, default=0.0, help='Z statistical filter k*std (0 disables)')
	p.add_argument('--stat-3d', type=float, default=0.0, help='3D radial filter k*std from centroid (0 disables)')
	p.add_argument('--max-n', type=int, default=None, help='Optional hard cap on point count after loading (random subset)')
	p.add_argument('--seed', type=int, default=123, help='Random seed for subsampling')
	return p.parse_args(argv)


def main(argv=None):
	args = parse_args(argv)
	pc = load_point_cloud(args.input)
	pc.ensure_float32()
	if args.max_n is not None and len(pc.xyz) > args.max_n:
		rng = np.random.default_rng(args.seed)
		idx = rng.choice(len(pc.xyz), args.max_n, replace=False)
		pc = PointCloud(pc.xyz[idx], pc.rgb[idx] if pc.rgb is not None else None)
	before = len(pc.xyz)
	pc = z_clip(pc, args.z_min, args.z_max)
	if args.stat_z > 0:
		pc = statistical_z(pc, args.stat_z)
	if args.stat_3d > 0:
		pc = statistical_radius(pc, args.stat_3d)
	after_filters = len(pc.xyz)
	chosen_voxel = args.voxel
	if args.target_n is not None and args.target_n > 0 and after_filters > 0:
		# Adaptive binary search over voxel size in XY bounding box extents.
		xy = pc.xyz[:, :2]
		min_xy = xy.min(axis=0)
		max_xy = xy.max(axis=0)
		span = max(max_xy - min_xy)
		# lower bound near zero (but not zero to avoid infinite loops), upper bound half the max span
		lo = span / 1e6 or 1e-6
		hi = max(span * 0.5, lo * 10)
		target = args.target_n
		best_pc = pc
		best_diff = float('inf')
		# limit iterations to avoid long runtimes
		for _ in range(25):
			mid = math.sqrt(lo * hi)  # geometric mean for smoother convergence
			test_pc = voxel_downsample_xy(pc, mid)
			n = len(test_pc.xyz)
			diff = abs(n - target)
			if diff < best_diff:
				best_diff = diff
				best_pc = test_pc
				chosen_voxel = mid
			if n > target:  # too many points -> increase voxel size
				lo = mid
			else:  # too few points -> decrease voxel size
				hi = mid
			# early break if within 1% or 50 points
			if diff / target < 0.01 or diff < 50:
				break
		pc = best_pc
	else:
		if chosen_voxel > 0:
			pc = voxel_downsample_xy(pc, chosen_voxel)
	after_voxel = len(pc.xyz)
	save_point_cloud(pc, args.output)
	print(f'Loaded {before} points -> after filters {after_filters} -> after voxel {after_voxel} (voxel={chosen_voxel:.6f}). Saved to {args.output}')


if __name__ == '__main__':
	main()

