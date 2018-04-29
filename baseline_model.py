#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 19:10:21 2017
@title: model manager
@author: shubham
"""
from custom_cost_functions import *
from keras.models import Model,load_model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D,concatenate
from keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint

#%%Model generator
def get_model (input_shape,
               unet_args={"lrate":1e-4,'momentum':0.9,"loss":'binary_crossentropy',"device":"/gpu:0"}):
    """Returns a U-net model."""
    print("="*8,"UNET","="*8)
    img_rows=input_shape[0]
    img_cols=input_shape[1]
    channels=input_shape[2]
    #%%Model architecture
    with K.tf.device(unet_args["device"]):
        inputs= Input((img_rows,img_cols,channels))
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        
        conv4b = Conv2D(256, (3, 3), activation='relu', padding='same')(pool4)
        conv4b = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4b)
        pool4b = MaxPooling2D(pool_size=(2, 2))(conv4b)
        
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4b)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
        
        up6b = UpSampling2D(size=(2, 2))(conv5)
        conv6b = Conv2D(256, (3, 3), activation='relu', padding='same')(up6b)
        conv6b= Conv2D(256, (3, 3), activation='relu', padding='same')(conv6b)
        
        up6 = UpSampling2D(size=(2, 2))(conv6b)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
        
        up7 = UpSampling2D(size=(2, 2))(conv6)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
        
        up8 = UpSampling2D(size=(2, 2))(conv7)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
        
        up9 = UpSampling2D(size=(2, 2))(conv8)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
        
        conv10 = Conv2D(4, (1, 1), activation='softmax')(conv9)
        
        model = Model(inputs=[inputs], outputs=[conv10])
        
        model.compile(optimizer=SGD(lr=unet_args['lrate'],momentum=unet_args['momentum']), loss=unet_args['loss'], metrics=['acc'])
        #%%Model summary:
        model.summary()
        print("input_shape",input_shape)
        print("optimizer:","SGD")
        print("unet_args:",unet_args)
        dummy=input("Press enter to proceed.")
    return model

def load_keras_model(path):
    return load_model(path)

print('-'*10,'baseline_unpooling_model.py loaded','-'*10)


if __name__=='__main__':
    m=get_model((64,64,2))
    