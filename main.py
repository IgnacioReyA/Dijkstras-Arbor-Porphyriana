from tin25d import build_tin
import numpy as np
import argparse
import os
import sys


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


if __name__ == "__main__":
    args = parse_args()
    save_tin(args.input, args.output_base)
