#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 16:18:21 2017
Keras Unpooling Layer
@author: shubham
"""
from read_activations import *
from default import *
np.set_printoptions(threshold=np.inf)
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Multiply
)

from keras.optimizers import SGD
from keras.layers.core import Lambda
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    Conv2DTranspose,
    UpSampling2D,
) 

from keras import backend as K
def _get_mask(**conv_params):
    if 'size' not in conv_params:
        conv_params['size'] = (2,2)
    if 'strides' not in conv_params:
        conv_params['strides']=None
    if 'padding' not in conv_params:
        conv_params['padding']= 'valid'
    if 'data_format' not in conv_params:
        conv_params['data_format']= None
    if 'pool_size' in conv_params:
        if size is not conv_params['size']:
            raise AssertionError('pool_size and size must be equal for\
                                 consistency in shapes.')
    else:
        conv_params['pool_size'] = conv_params['size']
    size=conv_params['size']
    pool_size = conv_params['pool_size']
    data_format = conv_params['data_format']
    padding=conv_params['padding']
    strides=conv_params['strides']
    #NOTE_ Ensure that the values provided are equal to those provided during
    #max_pooling
    
    def f(x,max_pooled_output=None):
        #TODO_ may use previously pooled output to save compute
        pooled_map = MaxPooling2D(pool_size=pool_size,strides=strides,
                                  padding=padding,data_format=data_format)(x)
        
        upsampled_map = UpSampling2D(size=size,
                                     data_format=data_format)(pooled_map)
        
        bool_mask = K.greater_equal(x, upsampled_map)
        mask = K.cast(bool_mask, dtype='float32')
#        print('Reached mask:',type(mask))
        return mask
    
    return Lambda(f)

def Unpool2D (**conv_params):
    if 'size' not in conv_params:
        conv_params['size'] = (2,2)
    if 'strides' not in conv_params:
        conv_params['strides']=None
    if 'padding' not in conv_params:
        conv_params['padding']= 'valid'
    if 'data_format' not in conv_params:
        conv_params['data_format']= None
        
    size=conv_params['size']
    data_format=conv_params['data_format']
    padding=conv_params['padding']
    strides=conv_params['strides']
    
    def f(x,tensor_to_get_indices):
        mask = _get_mask(size=size,strides=strides,padding=padding,
                         data_format=data_format)(tensor_to_get_indices)
        upsampled_x = UpSampling2D(size=size,
                                     data_format=data_format)(x)
        return Multiply()([upsampled_x,mask])
    
    return f


def get_model (shape=(256,256,1)):
    inp= Input(shape)
    conv1 = Conv2D(filters=1,kernel_size=(3,3),padding='same')(inp)
    mp1 = MaxPooling2D( pool_size=(2,2), padding = 'same')(conv1)
    unpool1= Unpool2D(size=(2,2),padding='same')(mp1,conv1)
    


#    inp= Input(shape)
#    mp1 = MaxPooling2D( pool_size=(2,2), padding = 'same')(inp)
#    conv1 = Conv2D(filters=2,kernel_size=(3,3),padding='same')(mp1)
#    mp2 = MaxPooling2D (pool_size=(2,2), padding = 'same')(conv1)
#    unpool1= Unpool2D(size=(2,2),padding='same')(mp2,conv1)
#    conv2 = Conv2D(filters=1,kernel_size=(3,3),padding='same')(unpool1)
#    unpool2= Unpool2D(size=(2,2),padding='same')(conv2,inp)
    
#    conv2 = Conv2D(filters=1,kernel_size=(3,3),padding='same')(unpool1)

    out = unpool1
#    mp1 = MaxPooling2D()(inp)
#    print('Inp type:',type(inp))
#    op1 = UpSampling2D()(mp1)
#    bool_mask = K.greater_equal(op1, inp)
#    print('Bool mask:',type(bool_mask))
##    mask =K.cast(bool_mask, dtype='float32')
#    mask =K.tensorflow_backend.cast(bool_mask, dtype='float32')
#    print(type(mask))
#    
##    x= K.tf.multiply(mask,inp)
#    x = Multiply()([mask, inp])
#    print('x:',type(x))
#    out = x
    model = Model(inputs=[inp],outputs=[out])
    model.compile(optimizer=SGD(lr=1e-3,momentum=0.9), loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    return model


def view_activation_maps(activations,names):
    z=activations
    for s in range(z[0].shape[0]):
        for i in range(len(z)):
            print('Shape:',z[i].shape)
            for j in range ( z[i].shape[3] ):
                print('j:',)
                win_name='s='+str(s)+' '+names[i]+' z['+str(i)+']['+str(j)+']'
                img=z[i][s,:,:,j:j+1]
                print(img.shape)
                print(win_name)
#                cv.namedWindow(win_name,cv.WINDOW_NORMAL)
                if np.max(img) <= 255:
                    cv.imshow(win_name,img.astype('uint8'))
                else:
                    print('For', win_name,'max=',np.max(img))
                    cv.imshow(win_name,(img/np.max(img)*255))
    cv.waitKey(0)
                    
def run_test_1():
    with hf.File(TRAIN_FOLDER+'/axial_with_less_bg.h5','r') as f:
        x=f['T1'][50:51]
        y=f['label'][50:51,:,:,1:2]
        m = get_model()
        m.fit(x,y,batch_size=10,epochs=0)
        m.save(OUTPUT_FOLDER+'/testing_unpool.h5')
        activations, names= get_activations(m,x,True)
        y=m.predict(x)
        view_activation_maps(activations,names)
        
#        cv.namedWindow('x',cv.WINDOW_NORMAL)
#        cv.namedWindow('y',cv.WINDOW_NORMAL)
#        cv.namedWindow('z',cv.WINDOW_NORMAL)
#        cv.imshow('x',x[0,:,:,0:1].astype('uint8'))
#        cv.imshow('y',y[0,:,:,0:1].astype('uint8'))
#        cv.imshow('z',z[0,:,:,0:1].astype('uint8'))
        print(np.unique(y))
        print('x__')
        print(np.unique(x))
        cv.waitKey(0)
        
def get_model_2 (shape=(256,256,1)):
    inp= Input(shape)
    out= MaxPooling2D((2,2),strides=(3,3), padding='same')(inp)
    
    model = Model(inputs=[inp],outputs=[out])
    model.compile(optimizer=SGD(lr=1e-3,momentum=0.9), loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    return model
    

if __name__ == '__main__':
    x = np.ones( (50,50), dtype = 'float32')
    y = np.zeros(shape=(101,101),dtype='float32')
    y[0:50,0:50]=x*0.
    y[0:50,50:100]=x*0.
    y[50:100,50:100]=x*0.
    y[50:100,0:50]=x*0.
    y[0:101,100:101]=255
    cv.imshow('x',y.astype('uint8'))
    cv.waitKey(0)
    y=y.reshape(1,101,101,1)
    
    l=[]
    for i in range(2):
        l.append(y)
    Y = np.concatenate(l,axis=0)
    m= get_model_2((101,101,1))
    act, names = get_activations(m,Y,True)
    view_activation_maps(act, names)
    
    
    

    
