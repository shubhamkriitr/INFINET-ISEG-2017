#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 18:01:48 2017

@author: shubham
"""
from UnpoolingLayer import UnPooling2D
from custom_cost_functions import *
from keras.models import Model,load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose,concatenate, BatchNormalization
from keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint
#%%Helpers
def _unpool_concat_block (input_layer,reference_1,reference_2,ref_depth,out_depth,**kwargs):
    unpool_1 = UnPooling2D(pool_size=(2,2),strides=(2,2))([input_layer, reference_1])
    concat_1 = concatenate ([unpool_1,reference_1])
    unpool_2 = UnPooling2D(pool_size=(2,2),strides=(2,2))([input_layer, reference_2])
    concat_2 = concatenate ([unpool_2,reference_2])
    concat = concatenate ([concat_1,concat_2])
#    conv = Conv2D(ref_depth, (3, 3), activation='relu', padding='same')(concat)
#    conv = BatchNormalization (axis = CHANNEL_AXIS) (conv)
    conv = Conv2D(out_depth, (1,1), activation='relu', padding='same')(concat)
    conv = BatchNormalization (axis = CHANNEL_AXIS) (conv)
    return conv

def _encoder_block (input_layer,filters_1,filters_2):
    conv = Conv2D(filters_1, (3, 3), activation='relu', padding='same')(input_layer)
    conv = BatchNormalization (axis = CHANNEL_AXIS) (conv)
    conv = Conv2D(filters_2, (3, 3), activation='relu', padding='same')(conv)
    conv = BatchNormalization (axis = CHANNEL_AXIS) (conv)
    return conv

def _decoder_block (input_layer,reference_1,reference_2,in_depth,out_depth):
    up1 = UnPooling2D()([input_layer, reference_1])
    up2 = UnPooling2D()([input_layer, reference_2])
    joint = concatenate ([up1, up2], axis= CHANNEL_AXIS)
    joint = Conv2D(in_depth, (1, 1), activation='relu',padding = 'same')(joint)
    concat = concatenate ([joint,reference_1,reference_2], axis = CHANNEL_AXIS)
    bn = BatchNormalization (axis = CHANNEL_AXIS) (concat)
    conv = Conv2D(out_depth, (3, 3), activation='relu', padding='same')(bn)
    conv = BatchNormalization (axis = CHANNEL_AXIS) (conv)
    return conv
    
#%%Model generator
def get_model (input_shape,unet_args={"lrate":1e-4,"loss":dice_coef_loss,"device":"/gpu:0"}):
    """Returns a CNN model."""
    print("="*8,"TWO IP MODALITY","="*8)
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
        #ENCODER_BRACNCH_1
        input_1= Input((img_rows,img_cols,channels))
        input_1_bn = BatchNormalization (axis = CHANNEL_AXIS) (input_1)
        conv1 = _encoder_block(input_1_bn,32,32)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
        conv2 = _encoder_block(pool1,64,64)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
        conv3 = _encoder_block(pool2,128,128)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
        conv4 = _encoder_block(pool3,256,256)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
        conv5 = _encoder_block(pool4,512,512)
        pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

        #ENCODER_BRANCH_2
        input_2 = Input((img_rows,img_cols,channels))
        input_2_bn = BatchNormalization (axis = CHANNEL_AXIS) (input_2)
        
        conv1_2 = _encoder_block(input_2_bn,32,32)
        pool1_2 = MaxPooling2D(pool_size=(2, 2))(conv1_2)
    
        conv2_2 = _encoder_block(pool1_2,64,64)
        pool2_2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)
    
        conv3_2 = _encoder_block(pool2_2,128,128)
        pool3_2 = MaxPooling2D(pool_size=(2, 2))(conv3_2)
    
        conv4_2 = _encoder_block(pool3_2,256,256)
        pool4_2 = MaxPooling2D(pool_size=(2, 2))(conv4_2)
    
        conv5_2 = _encoder_block(pool4_2,512,512)
        pool5_2 = MaxPooling2D(pool_size=(2, 2))(conv5_2)
        
        
        #BOTTLENECK
        up1 = UnPooling2D()([pool5, conv5])
        up2 = UnPooling2D()([pool5_2, conv5_2])
        bottle_neck = concatenate ([up1, up2], axis= CHANNEL_AXIS)
        bottle_neck = Conv2D(256, (1, 1), activation='relu',padding='same')(bottle_neck)
        bottle_neck = concatenate ([bottle_neck, up1, up2], axis = CHANNEL_AXIS)
        bottle_neck = BatchNormalization (axis = CHANNEL_AXIS) (bottle_neck)
        bottle_neck = Conv2D(256, (3, 3), activation='relu',padding='same')(bottle_neck)
        bottle_neck = BatchNormalization (axis = CHANNEL_AXIS) (bottle_neck)
        
        #DECODER_BLOCKS WITH UNPOOLING
        fuse_1 = _decoder_block(bottle_neck,conv4,conv4_2,256,128)
        fuse_2 = _decoder_block(fuse_1,conv3,conv3_2,128,64)
        fuse_3 = _decoder_block(fuse_2,conv2,conv2_2,64,32)
        fuse_4 = _decoder_block(fuse_3,conv1,conv1_2,32,32)
        
        #FINAL_SOFTMAX_LAYER
        conv10 = Conv2D(4, (1, 1), activation='softmax')(fuse_4)
    
        model = Model(inputs=[input_1,input_2], outputs=[conv10])
        model.compile(optimizer=SGD(lr=unet_args['lrate'],momentum=unet_args['momentum']), loss=unet_args['loss'], metrics=[dice_coef,dice_coef_loss,'acc'])
        #%%Model summary:
        model.summary()
        print("input_shape",input_shape)
        print("optimizer:","SGD")
        print("unet_args:",unet_args)
        dummy=input("Press enter to proceed.")
    return model

def load_keras_model(path):
    load_keras_model.custom_functions={'dice_coef_loss':dice_coef_loss,'UnPooling2D':UnPooling2D,
                      'dice_coef':dice_coef,'edge_loss':edge_loss,'mixed_loss':mixed_loss}
    return load_model(path,custom_objects=load_keras_model.custom_functions)

print('-'*10,'unet_with_sgd.py loaded','-'*10)


if __name__=='__main__':
    print(os.path.realpath("split_input_unet_sgd.py"))
    unet_args={'lrate':1e-4,'loss':'dice','momentum':0.9,'device':'/gpu:0'}
    get_model((256,256,1),unet_args)
