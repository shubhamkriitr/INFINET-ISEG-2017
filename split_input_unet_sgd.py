#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 19:10:21 2017
@title: model manager
@author: shubham
"""
from custom_cost_functions import *
from keras.models import Model,load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose,concatenate
from keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint

#%%Model generator
def get_model (input_shape,unet_args={"lrate":1e-4,"loss":dice_coef_loss,"device":"/gpu:0"}):
    """Returns a two-input-U-net model."""
    print("="*8,"UNET","="*8)
    img_rows=input_shape[0]
    img_cols=input_shape[1]
    if (input_shape[2] is not 1):
        print('Channel is assumed to be 1 and calculations',
              '\nwill be done accordingly.')
    channels=1
    
    if unet_args['loss']=='dice' or unet_args['loss']=='dice_coef_loss':
        unet_args['loss']=dice_coef_loss
    elif unet_args['loss']=='edge' or unet_args['loss']=='edge_loss':
        unet_args['loss']=edge_loss
    elif unet_args['loss']=='mixed' or unet_args['loss']=='mixed_loss':
        unet_args['loss']=mixed_loss
    
    #%%Model architecture
    with K.tf.device(unet_args["device"]):
        inputs= Input((img_rows,img_cols,channels))
        conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
        conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
        conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
        conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
        conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)
        
        inputs_2= Input((img_rows,img_cols,channels))
        conv1_2 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs_2)
        conv1_2 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1_2)
        pool1_2 = MaxPooling2D(pool_size=(2, 2))(conv1_2)
    
        conv2_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1_2)
        conv2_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2_2)
        pool2_2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)
    
        conv3_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2_2)
        conv3_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3_2)
        pool3_2 = MaxPooling2D(pool_size=(2, 2))(conv3_2)
    
        conv4_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3_2)
        conv4_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4_2)
        pool4_2 = MaxPooling2D(pool_size=(2, 2))(conv4_2)
    
        conv5_2 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool4_2)
        conv5_2 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5_2)
        
        
    
        up6 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5),
                           Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5_2),
                           conv4,conv4_2], axis=3)
        conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)
    
        up7 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6),conv3,conv3_2], axis=3)
        conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)
    
        up8 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv7),conv2,conv2_2], axis=3)
        conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv8)
    
        up9 = concatenate([Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv8),conv1,conv1_2], axis=3)
        conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv9)
    
        conv10 = Conv2D(4, (1, 1), activation='softmax')(conv9)
    
        model = Model(inputs=[inputs,inputs_2], outputs=[conv10])
        model.compile(optimizer=SGD(lr=unet_args['lrate'],momentum=unet_args['momentum']), loss=unet_args['loss'], metrics=[dice_coef,dice_coef_loss,'acc'])
        #%%Model summary:
        model.summary()
        print("input_shape",input_shape)
        print("optimizer:","SGD")
        print("unet_args:",unet_args)
        dummy=input("Press enter to proceed.")
    return model

def load_keras_model(path):
    load_keras_model.custom_functions={'dice_coef_loss':dice_coef_loss,
                      'dice_coef':dice_coef,'edge_loss':edge_loss,'mixed_loss':mixed_loss}
    return load_model(path,custom_objects=load_keras_model.custom_functions)

print('-'*10,'unet_with_sgd.py loaded','-'*10)


if __name__=='__main__':
    print(os.path.realpath("split_input_unet_sgd.py"))
    unet_args={'lrate':1e-4,'loss':'dice','momentum':0.9,'device':'/gpu:0'}
    get_model((256,256,1),unet_args)