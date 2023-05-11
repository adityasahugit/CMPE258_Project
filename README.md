# CMPE258_Project
Group project for Deep Learning Class- CMPE 258


**Group Members:-**

Aditya Sahu <br>
Divyam Sobti <br>
Prerna Shekar Bharadwaj <br>
Shashank Sharma


# How to run the project

#### Project Code Files Description:

In order to train the model, we have developed the code file “train_NN.py”. This file is responsible for training the model which is under the ‘training’ folder.

In order to run the model, we have developed the code file “detected_lanes_main.py”. This file takes input as a video file path and performs Lane Detection.


#### Steps for training the Model:- 

i) Download the dataset,  **full_CNN_train.p** and **full_CNN_labels.p** files from the below google drive link:- https://drive.google.com/drive/u/0/folders/1DlATz2oC-ui72HN66YDtkUvORjlDgL3f

ii) Execute the code through the below command.
```
python  train_NN.py
```

iii) Once the code runs, the model is saved as a CNN_model.h5 file. This model is used for detecting the lanes in detected_lanes_main.py file.


#### Steps for executing the Code:

i) For executing the program, pass “video_path” as parameter during running the program using the following command:-

```
python  detected_lanes_main.py –input video_path 
```

ii) Once the program is run, the file is saved as Output1.mp4



