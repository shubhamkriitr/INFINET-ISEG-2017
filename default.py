#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 20:20:08 2017

@author: shubham
"""
import os
import numpy as np
import nibabel as nib
import cv2 as cv
import h5py as hf

################DEFAULT PATHS######
print("-"*10,"default.py","-"*10)
SOURCE_CODE_ROOT_FOLDER=os.path.dirname(os.path.realpath(__file__))
ROOT_FOLDER=os.path.dirname(SOURCE_CODE_ROOT_FOLDER)

DATASET_FOLDER=os.path.join(ROOT_FOLDER,'Datasets')
TRAIN_FOLDER=os.path.join(DATASET_FOLDER,'Train')
TEST_FOLDER=os.path.join(DATASET_FOLDER,'Test')
MODEL_FOLDER=os.path.join(ROOT_FOLDER,'Models')
LOG_FOLDER=os.path.join(ROOT_FOLDER,'Logs')
OUTPUT_FOLDER=os.path.join(ROOT_FOLDER,'Outputs')

folders=[DATASET_FOLDER,TRAIN_FOLDER,TEST_FOLDER,
         MODEL_FOLDER,LOG_FOLDER,OUTPUT_FOLDER]
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)
##################IMAGE_DIMENSIONS############
NUM_OP_CHANNELS=4
CHANNELS=4
LENGTH=256
WIDTH=256
D_LENGTH=256#num of slices.
BACKEND='tensorflow'
IMAGE_DIM_ORDERING='tf'

if IMAGE_DIM_ORDERING=='tf':
    ROW_AXIS=1
    COL_AXIS=2
    CHANNEL_AXIS=3
elif IMAGE_DIM_ORDERING=='th':
    ROW_AXIS=2
    COL_AXIS=3
    CHANNEL_AXIS=1
##############################################

##########################################
CHANNEL_VALUES=[0,10,150,250]#DO N0T ALTER
#0->Background
#1->CSF
#2->Grey Matter
#3->White Matter
##########################################

##########################################
EXP_CHANNEL_VALUES=[0,255,0,0]#FOR TESTING
#0->Background
#1->CSF
#2->Grey Matter
#3->White Matter
##########################################
if __name__=='__main__':
    print(SOURCE_CODE_ROOT_FOLDER)
    print("ROOT:",ROOT_FOLDER)
    print(os.getcwd())
