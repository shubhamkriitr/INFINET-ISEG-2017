#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 01:55:56 2017

@author: shubham
"""
import train

print("script to test models")
print("Make sure to import the same network in train which is being tested.")


##FOR 64x64
#t=['micro_train.h5','micro_train.h5','micro_train.h5']
#v=None#['validation_2048_2559.h5']
#m='micro_7.h5'



#FOR 256x256
t = ['test_iseg_1.h5']#
c=['axial_with_less_bg_val.h5','coronal_with_less_bg.h5']
m='Y6_axis0_fold3.h5'
logb=m[:-3]+'_batchwise_60_record.h5'
loge=m[:-3]+'_epochwise_60_record.h5'
#train.util_obj.plot_history(train.util_obj.load_history(train.LOG_FOLDER+train.os.sep+logb),'batch')
train.util_obj.plot_history(train.util_obj.load_history(train.LOG_FOLDER+train.os.sep+loge),'epoch')

#train.util_obj.view_model_output(train.MODEL_FOLDER+"/"+m,
#                                train.TEST_FOLDER+"/"+t[0],num_of_inputs=1,normalize=False,scale=255)

#train.util_obj.compare_model_output(train.MODEL_FOLDER+"/"+m,
#                                train.TRAIN_FOLDER+"/"+c[0],num_of_inputs=2,normalize=False)