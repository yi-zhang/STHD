import copy
import json
import os

import anndata
import numpy as np
import squidpy as sq
import tifffile


class STHD:
    def __init__(self, spatial_path, counts_data, full_res_image_path, load_type):
        """Load ST-HD data.

        # When loading from original data source:
        STHD(
            spatial_path = '/hpc/group/yizhanglab/yz922/DATA/spatial/10x_HD_human_colon_cancer_20240325/square_002um/',
            counts_data = 'filtered_feature_bc_matrix.h5',
            full_res_image_path = '/hpc/group/yizhanglab/yz922/DATA/spatial/10x_HD_human_colon_cancer_20240325/Visium_HD_Human_Colon_Cancer_tissue_image.btf',
            load_type = 'original'
        )

        # When loading from a preloaded data source:
        STHD(
            spatial_path = adata,
            counts_data = None,
            full_res_image_path = '/hpc/group/yizhanglab/yz922/DATA/spatial/10x_HD_human_colon_cancer_20240325/Visium_HD_Human_Colon_Cancer_tissue_image.btf',
            load_type = 'preload'
        )

        # When loading from a cropped data source:
        STHD(
            spatial_path = '../analysis/20240331_hdcrop/crop1/adata.h5ad.gzip',
            counts_data = None,
            full_res_image_path = '../analysis/20240331_hdcrop/crop1/fullresimg_path.json',
            load_type = 'crop'
        )
        """
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
        """Crop in original full res image's pixel coordinates.

        (x1, y1) ... (x2, y1)
        .                   .
        .                   .
        (x1, y2) ... (x2, y2)

        factor: This is the scaling factor. by default we should use the scale factor for 'hires' image.

        Example:
        -------
        x1 = 57050
        y1 = 8600
        d = 200
        x2 = x1+d
        y2 = y1+d
        #fullresimg_subset = full_data.fullresimg[y1:y2, x1:x2,:]
        #plt.imshow(fullresimg_subset)
        crop_data = full_data.crop(
            x1, x2, y1, y2,
            full_data.adata.uns['spatial']['Visium_HD_Human_Colon_Cancer']['scalefactors']['tissue_hires_scalef']
        )
        crop_data.save('../analysis/20240331_hdcrop/crop4')

        """
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
        """Use reference gene expresssiong by cell type, and load to obj, adding a small noise.
        Params
        ----------
        ref: pd.DataFrame
            averaged gene expr from single cell reference, gene by celltype.

        Example:
        -------
        sec1_sthdata.adata.match_scrna_ref(genemeanpd_filtered)

        To get training dimensions:
        X = sec1_sthdata.adata.obs.shape[0] # n of spot
        Y = sec1_sthdata.adata.shape[1] # n of gene (filtered )
        Z = sec1_sthdata.lambda_cell_type_by_gene_matrix.shape[0]  # n of cell type

        """
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
        """x1 = 45450
        y1 = 4100
        d = 1100
        x2 = x1+d
        y2 = y1+d

        full_img = sthdata.load_img()
        adatacrop = sthdata.crop( x1, x2, y1, y2, factor = sthdata.adata.uns['spatial']['Visium_HD_Human_Colon_Cancer']['scalefactors']['tissue_hires_scalef'])
        x1c,x2c,y1c,y2c = adatacrop.get_sequencing_data_region()

        import matplotlib.pyplot as plt
        img_cropped = sthdata.crop_img(full_img,x1c,x2c,y1c,y2c)
        plt.imshow(img_cropped)
        """
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
