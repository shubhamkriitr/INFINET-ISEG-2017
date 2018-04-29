#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 11:57:25 2017

@author: shubham
"""
from default import *
from keras import backend as K
from keras import metrics
K.set_epsilon(1e-7)

#%% Handle dim ordering
assert(BACKEND==K.backend())
if BACKEND=='theano':
    K.theano.config.__setattr__('exception_verbosity','high')
    K.set_image_data_format('channels_first')
elif BACKEND=='tensorflow':
    K.set_image_data_format('channels_last')

print("Image_ordering:",K.image_dim_ordering())
print("Backend:",K.backend())

#%%LOSS FUNCTIONS

#%%1.DICE LOSS
smooth=1.0
w_dice=0.5

def old_dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) +
            smooth)

def dice_coef(y_true, y_pred):
    
    intersection=0
    #med_bal_factor=[210/16000,210/176,210/378,210/246]
    med_bal_factor=[ 0.04995437 , 1.73594757 , 0.82916348 , 1.25950102]
    union=0
    for i in range(4):
        intersection+= med_bal_factor[i]*(K.sum(y_true[:,:,:,i] * y_pred[:,:,:,i]))
        union+=med_bal_factor[i]*(K.sum(y_true[:,:,:,i]+y_pred[:,:,:,i]))
    return (2. * intersection + smooth) / (union + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
#%%2.MIXED_LOSS (MIXED DICE LOSS AND BINARY CROSSENTROPY LOSS) 
def mixed_loss (y_true,y_pred):
    return (w_dice*dice_coef_loss(y_true,y_pred)+(1-w_dice)*edge_loss(y_true,
                                                                      y_pred))
            
#            K.mean(K.binary_crossentropy(y_pred,
#            y_true), axis=-1))



#%%3.EDGE WEIGHTED LOSS
def get_edge_kernel():
    s=np.array([[1,  2,  -1],
                [2,  0, -2],
                [1, -2, -1]], dtype='float32')
    ker=np.zeros(shape=(3,3,4,4),dtype='float32')
    for i in range(4):
        ker[:,:,i,i]=s
    ker=K.constant(ker)
    return ker


KERNEL=get_edge_kernel()
def get_edge_tensor(img):
    return K.abs(K.conv2d(img,KERNEL,padding='same'))

def edge_loss(y_true,y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
    return - (K.tf.reduce_mean(K.tf.multiply(y_true * K.tf.log(y_pred),
                                             get_edge_tensor(y_true))))
    
#%%Testing module's functions
if __name__=='__main__':
    print(K.get_session())
    print('KERNEL:\n',KERNEL,"\n",KERNEL.eval(session=K.get_session()))
    print(K.get_session())