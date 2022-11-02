# Detect Cracks in a Wall for Building Inspections

## Overview

The goal of this project is the detection of defects in facades of buildings.
The idea behind the project is that you could fly over each side of a building and automatically detect the defects.
In this project Pytorch is used to train the VGG16 model with a Wall Crack Dataset from the Uta University.
In the next step the model validates if the video input has a wall crack in it or not. If there is a wall crack a mask gets overlayed over the video which marks the crack.
For performance enhancment only every 10th frame is used in the evaluation.

## Requirements

* Python => 3.10
* Python Libraries: `tkinter` `numpy` `matplotlib` `opencv-python` `torch` `torchvision`

### Database for model training

SDNET2018: A concrete crack image dataset for machine learning applications
https://digitalcommons.usu.edu/all_datasets/48/

## Run Detection

Run in directory (src\Detection)

```
python3 detection.py
```

## Sources

* Maguire, M., Dorafshan, S., & Thomas, R. J. (2018). SDNET2018: A concrete crack image dataset for machine learning applications. Utah State University. https://doi.org/10.15142/T3TD19