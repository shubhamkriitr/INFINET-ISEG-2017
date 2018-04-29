#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 18:01:48 2017

@author: shubham
"""
from UnpoolingLayer import UnPooling2D
from custom_cost_functions import *
from keras.models import Model,load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose,Concatenate, BatchNormalization
from keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint
#%%Helpers
FILTER_SHAPE = (5,5)
filter_shape = FILTER_SHAPE
def _dense_block (input_layer,f1,f2,f3):
    if 'count' not in _dense_block.__dict__:
        _dense_block.count = 0
    _dense_block.count+=1
    block_name = 'dns_'+str(_dense_block.count)+'_'
    suffix = ['conv1',#0
              'bn_conv1',#1
              'con_in_f1',#2
              'conv2',#3
              'bn_conv2',#4
              'concat_in_f1_f2',#5
              'conv1x1',#6
              'bn_conv1x1']#7
    
    names = []
    for i in range(len(suffix)):
        names.append(block_name+suffix[i])
    
    
    fmap1 = Conv2D(f1, filter_shape, activation='relu', padding='same',name=names[0])(input_layer)
    fmap1 = BatchNormalization (axis = CHANNEL_AXIS,name=names[1]) (fmap1)
    
    fmap2 = Concatenate (axis = CHANNEL_AXIS,name=names[2])([input_layer,fmap1])
    fmap2 = Conv2D(f2, filter_shape, activation='relu', padding='same',name=names[3])(fmap2)
    fmap2 = BatchNormalization (axis = CHANNEL_AXIS,name=names[4]) (fmap2)
    
    fmap3 = Concatenate ( axis = CHANNEL_AXIS,name=names[5]) ([input_layer,fmap1,fmap2])
    fmap3 = Conv2D(f3, (1,1), activation='relu', padding='same',name=names[6])(fmap3)
    out = BatchNormalization (axis = CHANNEL_AXIS,name=names[7]) (fmap3)
    
    return out

def encoder_block(input_layer,f1,f2,f3):
    return _dense_block(input_layer,f1,f2,f3)

def decoder_block(input_layer,f1,f2,f3):
    return _dense_block(input_layer,f1,f2,f3)

def concat_decoder_block(inp,ref1, ref2,f,f1,f2,f3):
    x = _unpool_concat_block(inp,ref1, ref2,f)
    out = _dense_block(x,f1,f2,f3)
    return out
    

def _bottle_neck (input_layer1,input_layer2,f):
    if 'count' not in _bottle_neck.__dict__:
        _bottle_neck.count = 0
    _bottle_neck.count+=1
    
    block_name = 'bo_'+str(_bottle_neck.count)+'_'
    suffix = ['conv1','bn_conv1',
              'conv2','bn_conv2',
              'concat','conv1x1',
              'bn_conv1x1']
    names=[]
    for i in range(len(suffix)):
        names.append(block_name+suffix[i])
    
    fmap1 = Conv2D(f, filter_shape, activation='relu', padding='same',name=names[0])(input_layer1)
    fmap1 = BatchNormalization (axis = CHANNEL_AXIS,name=names[1]) (fmap1)
    
    fmap2 = Conv2D(f, filter_shape, activation='relu', padding='same',name=names[2])(input_layer2)
    fmap2 = BatchNormalization (axis = CHANNEL_AXIS,name=names[3]) (fmap2)
    
    out = Concatenate(axis = CHANNEL_AXIS,name=names[4]) ([fmap1,fmap2])
    out = Conv2D(f, (1,1), activation='relu', padding='same',name=names[5])(out)
    out = BatchNormalization (axis = CHANNEL_AXIS,name=names[6]) (out)
    return out

def _unpool_concat_block (inp,ref1, ref2,f):
    if 'count' not in _unpool_concat_block.__dict__:
        _unpool_concat_block.count = 0
    _unpool_concat_block.count+=1
    block_name = 'upc'+str(_unpool_concat_block.count)+'_'
    
    suffix = ['up1','up2','concat','conv1x1','bn_conv1x1']
    names= []
    
    for i in range(len(suffix)):
        names.append(block_name + suffix[i])
    
    up1 = _unpooling_block(inp,ref1,name=names[0])
    up2 = _unpooling_block(inp,ref2,name=names[1])
    
    out = Concatenate ( axis = CHANNEL_AXIS,name=names[2]) ([up1,up2,ref1,ref2])
    out = Conv2D(f, (1,1), activation='relu', padding='same',name=names[3])(out)
    out = BatchNormalization (axis = CHANNEL_AXIS,name=names[4]) (out)
    return out

def _unpooling_block (input_layer,reference,name):
    return UnPooling2D(pool_size=(2,2),strides=(2,2),name=name)([input_layer, reference])

def _maxpooling_block(input_layer):
    if 'count' not in _maxpooling_block.__dict__:
        _maxpooling_block.count = 0
    _maxpooling_block.count +=1
    names= ['mp_'+str(_maxpooling_block.count)]
    return MaxPooling2D(pool_size=(2, 2),name = names[0])(input_layer)


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
        input_1= Input((img_rows,img_cols,channels),name='input_1')
        input_1_bn = BatchNormalization (axis = CHANNEL_AXIS,name='bn_input_1') (input_1)
        
        en1 = encoder_block(input_1_bn,64,64,64)
        pool1 = _maxpooling_block(en1)
        
        en2 = encoder_block(pool1,64,64,64)
        pool2 = _maxpooling_block(en2)
        
        en3 = encoder_block(pool2,64,64,64)
        
        
        #ENCODER_BRACNCH_2
        input_2= Input((img_rows,img_cols,channels), name= 'input_2')
        input_2_bn = BatchNormalization (axis = CHANNEL_AXIS, name='bn_input_2') (input_2)
        
        en1_2 = encoder_block(input_2_bn,64,64,64)
        pool1_2 = _maxpooling_block(en1_2)
        
        en2_2 = encoder_block(pool1_2,64,64,64)
        pool2_2 = _maxpooling_block(en2_2)
        
        en3_2 = encoder_block(pool2_2,64,64,64)
        
        #BOTTLENECK
        bottleneck = _bottle_neck(en3,en3_2,64)
        
        #DECODERS
        dec3 = decoder_block(bottleneck,64,64,64)
        dec2 = concat_decoder_block(dec3,en2,en2_2,64,64,64,64)
        dec1 = concat_decoder_block(dec2,en1,en1_2,64,64,64,64)
        
        
        #FINAL_SOFTMAX_LAYER
        conv10 = Conv2D(4, (1, 1), activation='softmax')(dec1)
    
        model = Model(inputs=[input_1,input_2], outputs=[conv10])
        model.compile(optimizer=SGD(lr=unet_args['lrate'],momentum=unet_args['momentum']), loss=unet_args['loss'],
                      metrics=[dice_coef,dice_coef_loss,dice_coef_0,
                                    dice_coef_1,dice_coef_2,dice_coef_3])
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


print('-'*10,'DENSE_NET_ADAPTED_3ENC','-'*10)


if __name__=='__main__':
    print(os.path.realpath("split_input_unet_sgd.py"))
    unet_args={'lrate':1e-4,'loss':'dice','momentum':0.9,'device':'/gpu:0'}
    get_model((256,256,1),unet_args)
