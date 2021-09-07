# Project - Robovid: keep social distance please!
This is a softare that can output the distance between two person in an image or video in real time.

If the distance of 2 person is below a threshold value (e.g. 1 metre), the program will output a message such as "Please social distance!" or "Keep X metres apart!" and a red box is drawn around those not social distancing.

![image](https://user-images.githubusercontent.com/83235099/132348354-554455b7-6f05-46e2-88a7-ede68868e292.png)
![image](https://user-images.githubusercontent.com/83235099/132348397-9f14c694-5d46-40e2-a031-446ef2557be5.png)

# Libraries to install first, enter into Command prompt:
`<pip install opencv-python>`

`<pip install matplotlib>`
# For Torchvision method 
`<pip install torchvision>`
# For Pixellib method
`<pip install tensorflow>`

`<pip install tensorflow-gpu>`

`<pip install pixellib>`

# How to use
1. First download "socialdistance.py" as contains all functions used for the program.
2. Decide which method you would like to use and download the corresponding python file. E.g. "pixellib image functions WORKING.py".
3. Run the selected python file and edit input value if required.

# To run Pixellib on GPU, follow the instructions to install NVIDIA software packages:
https://www.tensorflow.org/install/gpu

----------------------------------------
# System information
Platform:      Windows-10

Python:        3.7.0

PyCharm:       2021.1

Matplotlib:    3.4.4

NumPy:         1.19.5

OpenCV:        4.5.1
