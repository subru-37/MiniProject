# Number Plate Recognizing System

This repository contains the entire code used to developed a car plate detection system, from training and testing the model, to 
writing scripts to run in Raspberry Pi Model 4-B. First selecting an appropriate dataset from Kaggle, 
which includes images along with essential metadata like image dimensions and bounding box coordinates in .xml format. 
For model selection, after trying and testing various CNN architectures, Faster-RCNN combined with MobileNet V2 offered
the best balance between performance and memory requirements. Faster-RCNN has a significant performance boost due to
the integrated Region Proposal Network, while MobileNet V2 contributed higher resolution feature maps, enhancing accuracy. 
Hardware integration involved connecting a camera module via a MIPI CSI-2 port, downloading necessary libraries
using pip, enabling the camera through the configuration menu, and feeding captured images into the model for inference.

## Setting up the Raspberry Pi model 4-B

![image](https://github.com/subru-37/Number-Plate-Recognizing-System/assets/93091455/a1e95371-3eb3-4d2f-9440-c7c069d84596)

1. Insert the Micro SD Card with the Raspberry Pi OS image.
2. Connect the Camera module to the MIPI CSI - 2 connector.
3. Connect the required external output devices like monitor, keyboard and
mouse to the Raspberry Pi.
4. Directly connect the USB-C cable for power supply after ensuring there are
no loose connections.
5. After the desktop loads, run the code for model inference, image capture and
backend communication.

Note: Burn a Raspberry Pi OS Image using the instructions given in this [link](https://www.raspberrypi.com/software/).

## Pre-requisities

Install the below libraries for efficient working of code:
```
pip install pynput picamera2 numpy matplotlib torch torchvision pytesseract easyocr firebase-admin opencv-python
```

For the GPU version of pytorch, run:
```
pip uninstall torch torch audio torchvision
pip cache purge
pip install torch torchaudio torchvision -f https://download.pytorch.org/whl/torch_stable.html
```

## Output: 

![image](https://github.com/subru-37/Number-Plate-Recognizing-System/assets/93091455/5ea4541f-c44e-40f6-97ac-8aac93d1394b)
![image](https://github.com/subru-37/Number-Plate-Recognizing-System/assets/93091455/d4f9245c-972d-4d7f-b18a-ea8e74429174)
![image](https://github.com/subru-37/Number-Plate-Recognizing-System/assets/93091455/7fe84d34-0d7a-4c6c-9ab2-48009099653f)




