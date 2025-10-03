from tin25d import build_tin
import numpy as np
import argparse
import os
import sys
from tinyUtilities import barycentric_coords, interpolate_height, face_slope, aspect, point_in_triangle

# Build a TIN from a CSV and save the vertices, faces, and adjacency to separate CSV files
def save_tin(input_csv: str, output_base: str = "tin"):
    try:
        V, F, A = build_tin(input_csv)
    except FileNotFoundError:
        print(f"Error: input file not found: {input_csv}")
        print("Tip: the default sample is at 'Point Cloud/point_cloud.csv'.")
        sys.exit(1)
    except ImportError as e:
        print("ImportError:", e)
        print("SciPy is required for triangulation. Install with: pip install scipy")
        sys.exit(1)
    except ValueError as e:
        print("Input format error:", e)
        sys.exit(1)

    v_path, f_path, a_path = output_base + "_V.csv", output_base + "_F.csv", output_base + "_A.csv"
    np.savetxt(v_path, V, delimiter=",", header="x,y,z", comments="")
    np.savetxt(f_path, F, fmt="%d", delimiter=",", header="i0,i1,i2", comments="")
    np.savetxt(a_path, A, fmt="%d", delimiter=",", header="n0,n1,n2", comments="")
    print(f"Saved {v_path}, {f_path}, {a_path}")


# Parse command-line arguments for input CSV and output base filename
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build a 2.5D TIN and save CSVs")
    p.add_argument("-i", "--input", default=os.path.join("Point Cloud", "point_cloud.csv"), help="Input CSV with x,y,z columns")
    p.add_argument("-o", "--output-base", default="tin", help="Output base path (writes *_V.csv, *_F.csv, *_A.csv)")
    return p.parse_args(argv)

def test_tiny_utilities():
    # Define a 3D triangle (x, y, z)
    triangle = [(0, 0, 0), (1, 0, 0), (0, 1, 1)]
    point = (0.3, 0.3)  # point in XY plane

    print("=== Tiny Utilities Test ===")

    # 1 Barycentric coordinates
    u, v, w = barycentric_coords(point, triangle)
    print(f"Barycentric coords of {point}: u={u:.3f}, v={v:.3f}, w={w:.3f}")

    # 2 Interpolated height
    z = interpolate_height(point, triangle)
    print(f"Interpolated height at {point}: z={z:.3f}")

    # 3 Triangle normal (cross product of two edges)
    v0, v1, v2 = np.array(triangle[0]), np.array(triangle[1]), np.array(triangle[2])
    normal = np.cross(v1 - v0, v2 - v0)
    print(f"Triangle normal: {normal}")

    # 4 Face slope
    slope = face_slope(normal)
    print(f"Face slope: {slope:.2f}°")

    # 5 Aspect (orientation)
    asp = aspect(normal)
    print(f"Aspect: {asp:.2f}°")

    # 6 Point-in-triangle test
    inside = point_in_triangle(point, triangle)
    print(f"Point {point} inside triangle? {inside}")

if __name__ == "__main__":
    args = parse_args()
    save_tin(args.input, args.output_base)
    print("\n--- Testing geometric utilities ---")
    test_tiny_utilities()
    
    print("\n--- Testing geometric search queries ---")
    try:
        from geometricSearchQueries import test_geometric_search
        test_geometric_search()
    except ImportError as e:
        print(f"Could not import geometric search: {e}")
    except Exception as e:
        print(f"Error testing geometric search: {e}")