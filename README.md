# Itracker
## Introduction
This project is to learn about the process of gaze estimation. Among this project, the model is based on itracker and the dataset is ETH-Xgaze dataset.
## Requirements
* **Python** == 3.8
* **Torch** == 1.7.0+cu101
* **Visdom** == 0.1.8.9  
......  
More details in [requirements.txt](requirements.txt) file.
## File Structure
* **itracker** 
    * **itracker**   
    This directory contains the itracker model.
    * **itracker_mhsa**  
    This directory contains the itracker module and multi-head-self-attention module.
    * **prepareXgaze.py**   
    This file is used to detect right and left eyes.
    * **utils**  
    This directory contains some useful files to preprocess the dataset,evaluate the performance and so on.
* **data**  
This directory stores the preprocessed data and the checkpoints.And the detailed file structure is showed below.
```
├── data			
│   ├── xgaze_224
│   │   ├── train
│   │   │   ├── subject0000.h5
│   │   │   ├── ...
│   │   ├── test
│   │   │   ├── ...
│   ├── xgaze_landmarks
│   ├── train_eval_test_split.json
├── checkpoints
│   │   ├── botnet
│   │   ├── ...
├── src1
├── src2
├── requirements.txt
├── README.md
```
