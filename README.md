<p align="center">
  <h1 align="center">OxBrick A Dataset for High-fidelity<br>Object Reconstruction with Mobile Devices</h1>
  <p align="center">
    <a href="https://likojack.github.io/kejieli/#/home">Kejie Li</a>
    ·
    <a href="https://jwbian.net/">Jia-Wang Bian</a>
    ·
    <a href="https://scholar.google.co.uk/citations?user=ChYEbcsAAAAJ&hl=en">Robert Castle</a>
    ·
     <a href="https://torrvision.com/">Philip H.S. Torr</a>
    ·
     <a href="https://www.robots.ox.ac.uk/~victor/">Victor Adrian Prisacariu</a>   
  </p>

  <h3 align="center"><a href="">Project Page</a> | <a href="">arXiv</a> | <a href="">Dataset</a> </h3> 
  <div align="center"></div>
</p>

**How to generate high-quality 3D ground-truth shapes for reconstruction evaluation?**

<br>Even 3D scanners can only generate pseudo ground-truth shapes with artefacts.
OxBrick is the first **multi-view RGBD** dataset, captured on a **mobile device**, with **precise** 3D annotations for detailed 3D object reconstruction.

We propose a novel data capturing and 3D annotation pipeline in OxBrick without relying on expensive 3D scanners. 
The key to creating the precise 3D ground-truth shapes is using LEGO models, which are made of LEGO bricks with known geometry. 
The data modality of RGBD images captured on a mobile device paired with exact 3D geometry annotations provides a unique opportunity for future research on high-fidelity 3D reconstruction.


<!-- <p align="center">
    <img src="etc/teaser.png" alt="teaser" width="90%">
</p> -->

# Overview

1. [Install](#install)
1. [Our dataset](#dataset-organisation)
1. [Evaluation](#evaluation)
1. [Cite](#scroll-cite)
1. [License](#️page_with_curl-license)
1. [Changelog](#pencil-changelog)
1. [Acknowledgements](#octocat-acknowledgements)


# Install
you can install dependencies with Anaconda as follows: 
```shell
conda env create -f environment.yml
conda activate oxbrick
```

# Dataset Organisation
The dataset is organised by sequences, with xxx sequences of random shapes can be used for training, and xxx sequences of curated LEGO models for evaluation.

A sequence contains the following structure:
```

SEQUENCE_NAME
├── arkit_depth (the confidence and depth maps provided by ARKit)
|    ├── 000000_conf.png
|    ├── 000000.png
|    ├── 000000.npy
|    ├── ...
├── gt_depth (The high-resolution depth maps projected from the aligned GT shape)
|    ├── 000000.png
|    ├── 000000.npy
|    ├── ...     
├── image (the RGB images)
|    ├── 000000.jpg
|    ├── ...
├── mask (object foreground mask projected from the aligned GT shape)
|    ├── 000000.png
|    ├── ...
├── intrinsic (3x3 intrinsic matrix of each image)
|    ├── 000000.txt
|    ├── ...
├── pose (4x4 transformation matrix from camera to world of each image)
|    ├── 000000.txt
|    ├── ...
├── mesh
|    ├── high_res_tsdf_fusion.ply
|    ├── colmap_mesh.ply
|    ├── gt_points.ply
├── visibility_mask.npy (the visibility mask to be used for evaluation)
```

Note:
- the gt_points mesh is used for evaluation
- the high_res_tsdf_fusion is created by running tsdf-fusion using the gt depth

# Evaluation 
first preprocess the reconstruction by running preprocess_3d.py. This will crop the mesh to only visible parts using the visibility_mask.npy provided.
Then run evaluate_3d.py to get results.

```shell
python preprocess_3d.py
```
and evaluate it using
```shell
python evaluate_3d.py
```
A csv file with per-sequence results will be generated.


# Cite
Please cite our work if you find it useful or use any of our code
```latex
```

# ️License
Copyright © Niantic, Inc. 2022. Patent Pending. All rights reserved. This code is for non-commercial use. Please see the [license file](LICENSE) for terms.

# Changelog
- xx/03/2023: Dataset is online
