#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 05:54:31 2017

@author: shubham
"""
import train
import time
#%%With AUgmentation

#FOR 256x256
import imp
imp.reload(train)
#DATASET AND MODEL NAMES
tim = [time.time()]
train_subjects =[1,3,4,5,6,7,8,9,10]
val_subjects = [2]
def get_file_list (subjects,axis):
    prefix = 'LESS_BG/'+'axis'+str(axis)+'/subject-'
    l=[]
    for item in subjects:
        l.append(prefix+str(item)+'-axis-'+str(axis)+'_lbg.h5')
    return l
        
t = get_file_list(train_subjects,axis=0)
v=get_file_list(val_subjects,axis=0)
m='Y6_axis0_fold2.h5'

print('Training on:',t)
print('Validating on:',v)
#DATASET AND MODEL NAMES
names={'training_dataset_file_names': t  ,
            'validation_dataset_file_names': v,
            'output_model_file_name':  m}
##TRAINING PARAMS
epochs=70
batch_size=8
##NETWORK PARAMS
net_params={'lrate':1e-1,
            'momentum':0.95,
            'loss':'mixed',
            'num_of_inputs': 2,
            'device': '/gpu:1'}

##AUGMENTATION PARAMS
##DEFAULT PARAMS
args_dict={'featurewise_center':False,
            'samplewise_center':False,
            'featurewise_std_normalization':False,
            'samplewise_std_normalization':False,
            'zca_whitening':False,
            'rotation_range':10.0,
            'width_shift_range':0.2,
            'height_shift_range':0.2,
            'shear_range':None,#shear angle in radians
            'zoom_range':[0.9,1.05],
            'channel_shift_range':0.,
            'fill_mode':'nearest',
            'cval':0.,
            'horizontal_flip':True,
            'vertical_flip':True,
            'rescale':None,
            'preprocessing_function':None,
            'data_format':None,
            'seed':1,
            'num_of_input_branches':net_params['num_of_inputs'],
            'max_q_size':1,
            'elastic_deformation':False}

train.train_with_gen (names,batch_size=batch_size,epochs=epochs,interval=1,
            net_params=net_params,generator_args=args_dict)


tim.append(time.time())
#%%AXIS1
imp.reload(train)
t = get_file_list(train_subjects,axis=1)
v=get_file_list(val_subjects,axis=1)

print('Training on:',t)
print('Validating on:',v)
m='Y6_axis1_fold2.h5'
print(m*10)
print('THIS IS AN EXPERIMENTAL RUN TO SEE THE EFFECT OF ENSEMBLING. \
      MODEL USED IS UNET LIKE MODEL NOT THE DENSE ONE.')

#DATASET AND MODEL NAMES
names={'training_dataset_file_names': t  ,
            'validation_dataset_file_names': v,
            'output_model_file_name':  m}

train.train_with_gen (names,batch_size=batch_size,epochs=epochs,interval=1,
            net_params=net_params,generator_args=args_dict)

tim.append(time.time())
#%%AXIS2
imp.reload(train)
t = get_file_list(train_subjects,axis=2)
v=get_file_list(val_subjects,axis=2)

print('Training on:',t)
print('Validating on:',v)
m='Y6_axis2_fold2.h5'
print(m*10)

#DATASET AND MODEL NAMES
names={'training_dataset_file_names': t  ,
            'validation_dataset_file_names': v,
            'output_model_file_name':  m}
#
train.train_with_gen (names,batch_size=batch_size,epochs=epochs,interval=1,
            net_params=net_params,generator_args=args_dict)


tim.append(time.time())

print(tim)