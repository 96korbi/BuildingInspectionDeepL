# Building Inspection with Deep Learning
[![Licence](https://img.shields.io/github/license/Ileriayo/markdown-badges?style=for-the-badge)](./LICENSE.md)&nbsp;
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)&nbsp;
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)&nbsp;
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)](https://matplotlib.org/)&nbsp;
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)&nbsp;

## Overview

The goal of this project is the detection of defects in facades of buildings.
The idea behind the project is that you could fly over each side of a building and automatically detect the defects.
In this project Pytorch is used to train the VGG16 model with a Wall Crack Dataset from the Uta University.
In the next step the model validates if the video input has a wall crack in it or not. If there is a wall crack a mask gets overlayed over the video which marks the crack.
For performance enhancment only every 10th frame is used in the evaluation.

<div align="center">
  <img src="./static-files/testvideo.gif" alt="Test video showing different images of walls"></img>
</div>



## Requirements

* Python => 3.10
* Python Libraries: `tkinter` `numpy` `matplotlib` `opencv-python` `torch` `torchvision` `alive-progress`

### How to get the pretrained Model, not necessary if you want to train it yourself

It is safed with Git LFS so it won't be the right file when you run git clone
* Download it via Github => [Model](./src/Model/VGG16_v2-OCT_Building_half_dataset.pt)
* Get it via [Nextcloud](https://nextcloud.th-deg.de/s/RAZ4D4aNRns3Axo)

### Database for model training

[SDNET2018: A concrete crack image dataset for machine learning applications](https://digitalcommons.usu.edu/all_datasets/48/)

## Clone this repository

```
git clone https://github.com/96korbi/BuildingInspectionDeepL
```

On windows it could be neccessary to run this command as administrator to install PyTorch
```
pip install -r requirements.txt
```

Only neccessary for model training to visualize the images
```
apt install python3-tk
```

## Run detection

Run in directory (src\Detection)

```
python detection.py
```

## Run model training

Run in directory (src\Model)

```
python Pytorch.py
```

## Sources

* Maguire, M., Dorafshan, S., & Thomas, R. J. (2018). SDNET2018: A concrete crack image dataset for machine learning applications. Utah State University. https://doi.org/10.15142/T3TD19
