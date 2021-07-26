# Itracker
## Introduction
This project is to learn about the process of gaze estimation. Among this project, the model is based on itracker and the dataset is ETH-Xgaze dataset.
## Requirements
* **Python** == 3.8
* **Torch** == 1.7.0+cu101
* **Visdom** == 0.1.8.9   
More details in [requirements.txt](itracker/requirements.txt) file.
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
This directory stores the preprocessed data and the checkpoints. Detailed file structure is showed below.
```
├── data			
│   ├── train
│   │   ├── subject0000
│   │   │   ├── face
│   │   │   |   ├── 000000.jpg
│   │   │   |   ├── 000001.jpg
│   │   │   |   ├── ...  
│   │   │   ├── left_eye
│   │   │   |   ├── 000000.jpg
│   │   │   |   ├── 000001.jpg
│   │   │   |   ├── ... 
│   │   │   ├── right_eye
│   │   │   |   ├── 000000.jpg
│   │   │   |   ├── 000001.jpg
│   │   │   |   ├── ... 
│   │   ├── ...
│   ├── test
│   │   ├── subject0001
│   │   │   ├── face 
│   │   │   ├── left_eye
│   │   │   ├── right_eye
│   │   ├── ...
│   ├── val
│   │   ├── subject0003
│   │   │   ├── face 
│   │   │   ├── left_eye
│   │   │   ├── right_eye
│   │   ├── ...
│   ├── model_dir
│   │   ├── itracker
│   │   ├── itracker_mhsa
│   │   ├── ...
```
## Results
 model  | train_error  | val_error  | test_error  | test_error_std
 ---- | ----- | ------ | ------ | ------  
 itracker  | 4.895 | 6.8864 | 9.6374 | 9.1396 
 itracker_mhsa  | 3.7123 | 5.6807 | 8.3635 | 8.8631  
## Usage
Prepare the ETH-Xgaze dataset
'''
python prepareXgaze.py --dataset_path [source dataset path] --outer_dataset_path [the directory to store preprocessed data]
'''
Train itracker model
'''
python xgaze_main.py --data_path [preprocessed dataset path] --train True --load False
'''
Test itracker model
'''
python xgaze_main.py --data_path [preprocessed dataset path] --train False --load True
'''
