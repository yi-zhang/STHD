import copy
import json
import os

import anndata
import numpy as np
import squidpy as sq
import tifffile


class STHD:
    def __init__(self, spatial_path, counts_data, full_res_image_path, load_type):
        if load_type == "original":
            self.adata = sq.read.visium(path=spatial_path, counts_file=counts_data)
        elif load_type == "preload":
            self.adata = spatial_path
        elif load_type == "crop":
            self.adata = anndata.read_h5ad(spatial_path)
            with open(full_res_image_path) as f:
                full_res_image_path = json.load(f)["fullresimg_path"]

        self.fullresimg_path = full_res_image_path

    def crop(self, x1, x2, y1, y2, factor):
        # crop adata
        img = sq.im.ImageContainer.from_adata(
            self.adata
        )  # by default it got the first image, which is "hires"
        img_subset = img.crop_corner(
            int(y1 * factor),
            int(x1 * factor),
            (int((y2 - y1) * factor), int((x2 - x1) * factor)),
        )
        adata_subset = img_subset.subset(self.adata)

        # correct coordinates of the patch
        x1c, y1c, x2c, y2c = self.get_sequencing_data_region(adata_subset)
        print(f"Input Pixel is aligned to {x1c}, {y1c}, {x2c}, {y2c}")

        # crop fullresimg
        return STHD(adata_subset, None, self.fullresimg_path, "preload")

    def match_refscrna(
        self, ref, cutgene=True, gene_lambda_noise=0.000001, ref_renorm=False
    ):
        adata = self.adata
        adata.var_names_make_unique()
        overlap_gs = adata.var.index.intersection(ref.index)
        print(f"[Log] num of gene overlap {len( overlap_gs )}")
        if cutgene:
            adata = adata[:, adata.var_names.isin(overlap_gs)].copy()
        else:
            non_overlap_gs = [t for t in adata.var.index if t not in ref.index]
            adata = adata[:, ~non_overlap_gs]
        self.adata = adata

        ## add ref
        cell_type_by_gene_matrix = ref.loc[overlap_gs].T.values
        cell_type_by_gene_matrix = cell_type_by_gene_matrix + gene_lambda_noise
        if ref_renorm:
            print("[Log] normalizing lambda (not recommended)")
            cell_type_by_gene_matrix = (
                (cell_type_by_gene_matrix.T) / ((cell_type_by_gene_matrix.T).sum(0))
            ).T  # gene lambda for one cell type add up to 1.
        print("[Log] attaching ref gene expr in .lambda_cell_type_by_gene_matrix")
        self.lambda_cell_type_by_gene_matrix = cell_type_by_gene_matrix

    def copy(self):
        # deep copy
        new = copy.deepcopy(self)
        return new

    def load_img(self):
        img = tifffile.TiffFile(self.fullresimg_path).asarray()
        return img

    def crop_img(self, img, x1, y1, x2, y2):
        img_cropped = img[y1:y2, x1:x2, :]
        return img_cropped

    def get_sequencing_data_region(self, adata=None):
        if not adata:
            adata = self.adata
        x1, y1 = np.nanmin(adata.obsm["spatial"], axis=0)
        x2, y2 = np.nanmax(adata.obsm["spatial"], axis=0)
        x1, y1, x2, y2 = [int(i) for i in [x1, y1, x2, y2]]
        return x1, y1, x2, y2

    def save(self, path):
        """Save data."""
        # create path if not exist
        if not os.path.exists(path):
            print("creating new folder to save cropped data: ", path)
            os.makedirs(path)

        # save adata
        adata = self.adata
        adata.write_h5ad(os.path.join(path, "adata.h5ad.gzip"), compression="gzip")

        # save fullresimg_path
        with open(os.path.join(path, "fullresimg_path.json"), "w+") as f:
            json.dump({"fullresimg_path": self.fullresimg_path}, f, indent=4)
