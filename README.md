<p align="center">
  <h1 align="center">Building LEGO for 3D Reconstruction on Mobile Devices</h1>
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

  <h3 align="center"><a href="https://code.active.vision/MobileBrick/">Project Page</a> | <a href="https://arxiv.org/abs/2303.01932">arXiv</a> | <a href="http://www.robots.ox.ac.uk/~victor/data/MobileBrick/MobileBrick_Mar23.zip">Dataset</a> </h3> 
  <div align="center"></div>
</p>


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
conda activate mobilebrick
```

# Dataset Organisation
The dataset is organised by sequences, with 135 sequences of random shapes can be used for training, and 18 sequences of manually curated LEGO models for evaluation.

A sequence contains the following structure:
```

SEQUENCE_NAME
├── arkit_depth (the confidence and depth maps provided by ARKit)
|    ├── 000000_conf.png
|    ├── 000000.png
|    ├── ...
├── gt_depth (The high-resolution depth maps projected from the aligned GT shape)
|    ├── 000000.png
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
|    ├── gt_mesh.ply
├── visibility_mask.npy (the visibility mask to be used for evaluation)
├── cameras.npz (processed camera poses using the format of NeuS)
```

Note:
- the gt_mesh.ply is created by running tsdf-fusion using the gt depth

# Evaluation 
We provide scripts to run evaluation on 3D reconstruction and Novel View Synthesis (NVS).

To evaluate 3D reconstruction, use the following code.
```
python evaluations/evaluate_3d.py --method $METHOD
```
The reconstruction files (.ply) to be evaluated should be places in the ```./meshes/$METHOD``` folder. A .csv file with per-sequence results will be generated.

To evaluate NVS, use the following code.
```shell
python evaluate_nvs.py --method $METHOD
```
The rendered images for evaluation should be placed in ```./nvs/$METHOD```


# Cite
Please cite our work if you find it useful or use any of our code
```latex
@article{li2023mobilebrick,
  author = {Kejie Li, Jia-Wang Bian, Robert Castle, Philip H.S. Torr, Victor Adrian Prisacariu},
  title = {MobileBrick: Building LEGO for 3D Reconstruction on Mobile Devices},
  journal={arXiv preprint arXiv:2303.01932},
  year={2023}
}
```

# Changelog
- 06/03/2023: Dataset is online
