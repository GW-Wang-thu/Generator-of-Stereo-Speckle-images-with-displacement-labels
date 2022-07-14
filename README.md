# Generator-of-stereo-speckle-images-with-displacement-labels
A stereo speckle images generator with nonlinear full-field displacement based on Newton iteration. Used for the research of 3D-DIC or other scene flow estimation tasks.

## Introduction
- A stereo speckle images generator based on stereo vision theory
- With nonlinear 3D displacement labels, time-wise optical flow labels and right-to-left disparity labels.
- Digit twin of an arbitrary undistorted stereo imaging system(with calibration files)
- Also can be used to generate speckle image paires with pixel displacement, i.e., the left reference and the left current images with the optical flow label.
- For algorithm details and the principles, please refer to ***(unpublished work, update later)

## Requests
- python38
- opencv-python (4.4.0 used)
- numpy (1.22.1 used)
- pytorch and torchvision with cuda (torch1.9.0+torchvision0.10.0+cu111)
- (optional) numba

## Workflow
* (Optional) Give your stereo calibration parameters and other settings in ```./Seeds/input.py```
* (Optional) Generate your own seeds by running ```input.py```
* Run ```main_box.py```

## Sample
- Dataset sample can be download from [Google Drive](https://drive.google.com/drive/folders/1vhRsQilTJcGXLwSiknJA7hgsFPOIXPo_?usp=sharing)

- The speckle images sample:
![The speckle images](/imgs/speckle_images.png)

- The 3D displacement labels and the results calculated using 3D-DIC (mm)
![The 3D displacement labels and the results calculated using 3D-DIC](/imgs/UVW.png)

- The optical flow and disparity labels and that calculated using 2D-DIC (Pixels):
![The optical flow and disparity labels and that calculated using 2D-DIC](/imgs/flow_disparity.png)


## Cite this work
```
@article{WANG2022107184,
title = {StrainNet-3D: Real-time and robust 3-dimensional speckle image correlation using deep learning},
journal = {Optics and Lasers in Engineering},
volume = {158},
pages = {107184},
year = {2022},
issn = {0143-8166},
doi = {https://doi.org/10.1016/j.optlaseng.2022.107184},
url = {https://www.sciencedirect.com/science/article/pii/S0143816622002378},
author = {Guowen Wang and Laibin Zhang and Xuefeng Yao}
}
```
