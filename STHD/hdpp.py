import os

import pandas as pd
import argparse

def format_tissue_positions(spatial_dir):
    tissue_parquet_path = os.path.join(spatial_dir, "tissue_positions.parquet")
    out_file = os.path.join(spatial_dir, "tissue_positions.csv")
    if os.path.isfile(out_file):
        print("File exist", out_file)
    position_parquet = pd.read_parquet(tissue_parquet_path)
    res = position_parquet.set_index("barcode").applymap(lambda t: round(t))
    res.to_csv(out_file, sep=",", header=False, index=True)
    print("Output in ", os.path.join(spatial_dir, "tissue_positions.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spatialdir",
        type=str,
        help="spatial directory under square_002um",
    )
    args = parser.parse_args()
    print(f'[Log] converting tissue position in {args.spatialdir}')
    format_tissue_positions(args.spatialdir)