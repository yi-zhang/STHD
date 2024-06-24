# STHD: probabilistic cell typing of Single spots in whole Transcriptome spatial data with High Definition

---

- Quick start: `notebooks/tutorial.ipynb`
- Generates single-spot (2um) cell type labels and probabilities for VisiumHD data using a machine learning model.
- Input: VisiumHD data and reference scRNA-seq dataset with cell type annotation.
- Output: cell type labels and probabilities at 2um spot level.
- Visualization - STHDviewer: interactive, scalable, and fast spatial plot of spot cell type labels, in a HTML.

---

- Author: Yi Zhang, PhD, yi.zhang@duke.edu
- Website: [Yi Zhang Lab at Duke](https://yi-zhang-compbio-lab.github.io)
- STHDviewer of VisiumHD colon cancer sample with near 9 million spots: [STHDviewer_colon_cancer_HD](https://yi-zhang-compbio-lab.github.io/STHDviewer_colon_cancer_hd)
- We provided [test data](https://duke.box.com/v/yi-zhang-duke-sthd-test). Download this folder and put as `./testdata/`

---
- python version requirement: >=3.8.3
- How to use
  - create new python venv `python3.8 -m venv sthd_env`
  - activate the venv `source sthd_env/bin/activate`
  - download repo: `git clone git@github.com:yi-zhang/STHD.git`
  - install dependencies: `pip install -r STHD/requirements.txt`
  - making sure `./STHD` is in python path, e.g adding via `sys.path.append('./STHD')`
  - then in script: `from STHD import {the module you need}`
- Beta version - pip package coming soon; also finalizing details of comprehensive tutorials! 

## STHD Quickstart using a colon cancer VisiumHD patch:

- See `notebooks/tutorial.ipynb`
- The test data includes a patch crop from the VisiumHD file in `testdata/crop10`
  
## (optional) Preparing VisiumHD sample. 
- 10X Genomics colon cancer sample can be downloaded from: https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-human-crc
- Required input includes 2um level spatial expression: `square_002um` , which usually contains filtered_feature_bc_matrix.h5 and spatial/tissue_positions.csv . It is often from the downloaded folder "Binned outputs (all bin levels)". tissue positions in .parquet format can be converted using STHD/hdpp.py.
- Required input also includes full-resolution H&E image: Visium_HD_Human_Colon_Cancer_tissue_image.btf. It is often from the "Microscope image".

## References
- bioRxiv link coming
