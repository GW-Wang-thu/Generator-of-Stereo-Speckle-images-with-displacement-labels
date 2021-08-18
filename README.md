# Generator-of-Stereo-Speckle-images-with-displacement-labels
A stereo speckle images generator based on Newton iteration for any nonlinear displacement, for the research of 3D DIC or other scene flow estimation tasks.

## Workflow

* (Optional) Give calibration parameters in ```input.py```
* (Optional) Generate your own seeds by running ```input.py```
* Change the input seed file in line 397 in ```main.py``` 
* Run ```main.py``` to generate your dataset.

## Sample

* A gif of the four speckle images:

![Speckle Images](https://github.com/GW-Wang-thu/Generator-of-Stereo-Speckle-images-with-displacement-labels/tree/main/Sample/Speckle_Images.gif)

* The Optical Flow and Disparity labels, compared with the results calculated by 2D-DIC:

![Optical Flow and Disparity](https://github.com/GW-Wang-thu/Generator-of-Stereo-Speckle-images-with-displacement-labels/tree/main/Sample/FlowDisp.png)

* The Displacement U, V and W labels, compared with the results calculated by 3D-DIC:

![Displacement U, V and W](https://github.com/GW-Wang-thu/Generator-of-Stereo-Speckle-images-with-displacement-labels/tree/main/Sample/UVW.png)

** Note: You can use the initial images in ```/Samples``` to calculate the above results. **

## Cite this work



