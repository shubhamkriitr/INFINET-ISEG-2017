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
def _encoder_block (input_layer,filters_1,filters_2):
    conv = Conv2D(filters_1, (3, 3), activation='relu', padding='same')(input_layer)
    conv = BatchNormalization (axis = CHANNEL_AXIS) (conv)
    conv = Conv2D(filters_2, (3, 3), activation='relu', padding='same')(conv)
    conv = BatchNormalization (axis = CHANNEL_AXIS) (conv)
    return conv

def _bottle_neck (first_branch,second_branch,first_ref,second_ref,f,g):
    first_up = UnPooling2D()([first_branch,first_ref])
    second_up = UnPooling2D() ([second_branch,second_ref])
    
    concat_unpooled = concatenate ([first_up,second_up],axis = CHANNEL_AXIS)
    concat_unpooled_bn = BatchNormalization (axis = CHANNEL_AXIS) (concat_unpooled)
    
    conv1x1 = Conv2D(f, (1, 1), activation='relu', padding='same')(concat_unpooled_bn)
    conv1x1bn = BatchNormalization (axis = CHANNEL_AXIS) (conv1x1)
    
    all_concat = concatenate([conv1x1bn,first_ref,second_ref])
    all_concat_bn = BatchNormalization (axis = CHANNEL_AXIS) (all_concat)
    
    conv3x3 = Conv2D(g, (3, 3), activation='relu', padding='same')(all_concat_bn)
    conv3x3_bn = BatchNormalization (axis = CHANNEL_AXIS) (conv3x3)
    
    return conv3x3_bn

def _decoder_block (input_layer,first_ref,second_ref,f,g):
    return _bottle_neck (input_layer,input_layer,first_ref,second_ref,f,g)

def _maxpool (inp_layer):
    return MaxPooling2D(pool_size=(2, 2))(inp_layer)
    


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
        
        branch1_en1 = _encoder_block(input_1_bn,64,64)
        branch1_mp1 = _maxpool(branch1_en1)
        
        branch1_en2 = _encoder_block(branch1_mp1,64,64)
        branch1_mp2 = _maxpool(branch1_en2)
        
        branch1_en3 = _encoder_block(branch1_mp2,64,64)
        branch1_mp3 = _maxpool(branch1_en3)
        

        #ENCODER_BRANCH_2
        input_2 = Input((img_rows,img_cols,channels))
        input_2_bn = BatchNormalization (axis = CHANNEL_AXIS) (input_2)
        
        b2_en1 = _encoder_block(input_2_bn,64,64)
        b2_mp1 = _maxpool(b2_en1)
        
        b2_en2 = _encoder_block(b2_mp1,64,64)
        b2_mp2 = _maxpool(b2_en2)
        
        b2_en3 = _encoder_block(b2_mp2,64,64)
        b2_mp3 = _maxpool(b2_en3)
        
        #BOTTLENECK
        bottle_neck = _bottle_neck(branch1_mp3,b2_mp3,branch1_en3,b2_en3,64,64)
        
        #DECODER_BLOCKS WITH UNPOOLING
        dec_2 = _decoder_block(bottle_neck,branch1_en2,b2_en2,64,64)
        dec_1 = _decoder_block(dec_2,branch1_en1,b2_en1,64,64)
        
        
        #FINAL_SOFTMAX_LAYER
        conv10 = Conv2D(4, (1, 1), activation='softmax')(dec_1)
    
        model = Model(inputs=[input_1,input_2], outputs=[conv10])
        model.compile(optimizer=SGD(lr=unet_args['lrate'],momentum=unet_args['momentum']), loss=unet_args['loss'],
                      metrics=[dice_coef,dice_coef_loss,dice_coef_0,
                                    dice_coef_1,dice_coef_2,dice_coef_3,'acc'])
        #%%Model summary:
        model.summary()
        print("input_shape",input_shape)
        print("optimizer:","SGD")
        print("unet_args:",unet_args)
        dummy=input("Press enter to proceed.")
    return model

def load_keras_model(path):
    load_keras_model.custom_functions={'dice_coef_loss':dice_coef_loss,'UnPooling2D':UnPooling2D,
                      'dice_coef':dice_coef,'edge_loss':edge_loss,'mixed_loss':mixed_loss,
                      'dice_coef_0':dice_coef_0,'dice_coef_1':dice_coef_1,'dice_coef_2':dice_coef_2,
                      'dice_coef_3':dice_coef_3,'dice_coef_axis':dice_coef_axis}
    return load_model(path,custom_objects=load_keras_model.custom_functions)

print('-'*10,'MODEL_Y(With 3 Encoders/Decoders)','-'*10)


if __name__=='__main__':
    print(os.path.realpath("split_input_unet_sgd.py"))
    unet_args={'lrate':1e-4,'loss':'dice','momentum':0.9,'device':'/gpu:0'}
    get_model((256,256,1),unet_args)
