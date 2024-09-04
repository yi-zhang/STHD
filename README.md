# STHD: probabilistic cell typing of Single spots in whole Transcriptome spatial data with High Definition
<img width="862" alt="sthd_git1fig" src="https://github.com/user-attachments/assets/2477197e-888f-4383-b5cd-e7e256eefbc0">
---

- Quick start: `notebooks/tutorial.ipynb`
- Generates single-spot (2um) cell type labels and probabilities for VisiumHD data using a machine learning model.
- Input: VisiumHD data and reference scRNA-seq dataset with cell type annotation.
- Output: cell type labels and probabilities at 2um spot level.
- Visualization - STHDviewer: interactive, scalable, and fast spatial plot of spot cell type labels, in a HTML.


---

- Author: Yi Zhang, PhD, yi.zhang@duke.edu
- Website: [Yi Zhang Lab at Duke](https://yi-zhang-compbio-lab.github.io)
- STHDviewer of VisiumHD colon cancer sample with near 9 million spots: STHDviewer_colon_cancer_HD:[https://yi-zhang-compbio-lab.github.io/STHDviewer_colon_cancer_hd](https://yi-zhang-compbio-lab.github.io/STHDviewer_colon_cancer_hd/STHDviewer_crchd.html)
- We provided [test data](https://duke.box.com/v/yi-zhang-duke-sthd-test). Download this folder and put as `./testdata/`

## Install
---
- python version requirement: >=3.8.3
- How to use
  - create new python venv `python3.8 -m venv sthd_env`
  - activate the venv `source sthd_env/bin/activate`
  - download repo: `git clone git@github.com:yi-zhang/STHD.git`
  - install dependencies: `pip install -r STHD/requirements.txt`
  - making sure `./STHD` is in python path, e.g adding via `sys.path.append('./STHD')`
  - then in script: `from STHD import {the module you need}`
- pip package coming soon

## STHD Quickstart using a colon cancer VisiumHD patch:

- See `notebooks/tutorial.ipynb`
- The test data includes a patch crop from the VisiumHD file in `testdata/crop10`

---

## STHD pipeline on a larger VisiumHD region, or the full VisiumHD sample:

### Step 1: prepare normalized gene expression profile (lambda) by cell type from reference scRNA-seq data.

- This step will generate the reference file. Details are in  `notebooks/s01_build_ref_scrna.ipynb`
- We provided the processed file `./testdata/crc_average_expr_genenorm_lambda_98ct_4618gs.txt`

### Step 2: pre-processing of VisiumHD data files

- The test data includes a larger region from the VisiumHD file in `testdata/crop10large/`
  
#### Preparing the VisiumHD sample. 
- 10X Genomics colon cancer sample can be downloaded from: https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-human-crc
- Required input includes 2um level spatial expression: `square_002um` , which usually contains filtered_feature_bc_matrix.h5 and spatial/tissue_positions.csv . It is often from the downloaded folder "Binned outputs (all bin levels)". tissue positions in .parquet format can be converted using STHD/hdpp.py.
- Required input also includes full-resolution H&E image: Visium_HD_Human_Colon_Cancer_tissue_image.btf. It is often from the "Microscope image".
- The scale factor number will also be useful, which is usually in square_002um/spatial/scalefactors_json.json
- Our processed data files are available as in: `testdata/VisiumHD/`

### Step 3: Patchify the large region

- This step will take a large region and split into patches. Details are in `notebooks/s11_patchify.ipynb`
- Or, use example command line:

```bash
# Spliting patches from a test large cropped data:
python3 -m STHD.patchify \
--spatial_path ./testdata/crop10large/all_region/adata.h5ad.gzip \
--full_res_image_path ./testdata/crop10large/all_region/fullresimg_path.json \
--load_type crop \
--dx 1500 \
--dy 1500 \
--scale_factor 0.07973422 \
--refile ./testdata/crc_average_expr_genenorm_lambda_98ct_4618gs.txt \
--save_path ./testdata/crop10large_patchify \
--mode split
```

- For full sample, example command line below (will take some space and time)

```bash
# Spliting patches from the full-size VisiumHD sample:
python3 -m STHD.patchify \
--spatial_path ./testdata/VisiumHD/square_002um/ \
--counts_data filtered_feature_bc_matrix.h5 \
--full_res_image_path ./testdata/VisiumHD/Visium_HD_Human_Colon_Cancer_tissue_image.btf \
--load_type original \
--dx 6000 \
--dy 6000 \
--scale_factor 0.07973422 \
--refile ./testdata/crc_average_expr_genenorm_lambda_98ct_4618gs.txt \
--save_path ./analysis/full_patchify \
--mode split
```

### Step 4: Obtain training command line for the patch list

- This step trains STHD on each patch. The command can be flexibly modified to submit to different slurm jobs on a HPC. Details are in  `notebooks/s12_per_patch_train.ipynb` ,Or,
- Example command is:

```bash
python3 -m STHD.train --refile ./testdata/crc_average_expr_genenorm_lambda_98ct_4618gs.txt \
--patch_list ./testdata/crop10large/patches/52979_9480 ./testdata/crop10large/patches/57479_9480 ./testdata/crop10large/patches/52979_7980 ./testdata/crop10large/patches/55979_7980 ./testdata/crop10large/patches/57479_7980 ./testdata/crop10large/patches/54479_9480 ./testdata/crop10large/patches/55979_9480 ./testdata/crop10large/patches/54479_7980
```

### Step 5: Combine the patch results

- This step combines STHD patch-wise results together. Details are in  `notebooks/s13_combine_patch.ipynb`, Or

```bash
#Combine predictions
python3 -m STHD.patchify \
--refile ./testdata/crc_average_expr_genenorm_lambda_98ct_4618gs.txt \
--save_path ./testdata/crop10large_patchify \
--mode combine
```

### Step 6: Visualize!

- This step takes STHD results on a large region and generate STHDviewer for interactive exploration. Details are in `notebooks/s21_visualize.ipynb`

### Step 7: Downstream analyses

- One example is STHD-guided binning using a size of choice for —nspot
- Details are in `notebooks/s04_STHD_cell_type_guided_binning.ipynb`; Or.

```bash
python -m STHD.binning_fast --patch_path ./testdata/crop10/ --nspot 4 --outfile ./testdata/crop10_STHDbin_nspot4.h5ad
```

---

## Dependencies

please check requirements.txt
