#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 12:31:12 2017

@author: shubham
"""
from read_activations import *
from default import *
#np.set_printoptions(threshold=np.inf)
from keras.models import Model
from keras.layers import Layer
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
    UpSampling2D
) 
from keras.utils import conv_utils
from keras.engine import InputSpec
from keras import backend as K

class UnPooling2D (Layer):
    """Abstract class for different pooling 2D layers.
    """

    def __init__(self, pool_size=(2, 2), strides=None, padding='same',
                 data_format=None, **kwargs):
        super(UnPooling2D, self).__init__(**kwargs)
        data_format = conv_utils.normalize_data_format(data_format)
        if strides is None:
            strides = pool_size
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4)]
        self.expected_out_shape= None
        self.size = None
        if padding is not 'same':
            print('This Layer is implemented only\
            for padding="same".')
        if pool_size!=strides:
            print('pool_size:',pool_size,'strides:',strides)
            raise ValueError ('Keep strides and pool_size equal.')

    def compute_output_shape(self, input_shapes):
        return input_shapes[1]
    
    def _compute_pool_output_shape(self, input_shapes):
        input_shape = input_shapes[1]
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
        rows = conv_utils.conv_output_length(rows, self.pool_size[0],
                                             self.padding, self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.pool_size[1],
                                             self.padding, self.strides[1])
        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1], rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, input_shape[3])
    
    
    def _get_op_rows_cols (self):
        if self.data_format=='channels_first':
            return (self.expected_out_shape[2],self.expected_out_shape[3])
        elif self.data_format == 'channels_last':
            return (self.expected_out_shape[1],self.expected_out_shape[2])
    def _get_rows_cols (self,shape):
        if self.data_format=='channels_first':
            return (shape[2],shape[3])
        elif self.data_format == 'channels_last':
            return (shape[1],shape[2])
    
    def _adjust_size (self, input_):
        rows,cols = self._get_op_rows_cols()
        if self.data_format=='channels_first':
            return input_[:,:,0:rows,0:cols]
        elif self.data_format == 'channels_last':
            return input_[:,0:rows,0:cols,:]

    def _unpooling_function(self, inputs, pool_size, strides,
                          padding, data_format):
        return self._op_when_equal(inputs, pool_size, strides,
                          padding, data_format)
    
    def _op_when_equal(self, inputs, pool_size, strides,
                          padding, data_format):
        self.expected_out_shape = K.int_shape(inputs[1])
        self.size = self.strides
        mask = K.pool2d(inputs[1], pool_size, strides,
                          padding, data_format,
                          pool_mode='max')
        mask = K.resize_images(mask,self.size[0],self.size[1],
                                                self.data_format)
        adjusted_ref = self.adjust_reference_tensor(inputs[1],case='equal')
        mask = K.greater_equal(adjusted_ref, mask)
        mask = self._adjust_size(mask )
        mask = K.cast(mask, dtype='float32')
        upsampled_input = self._adjust_size(K.resize_images(inputs[0], self.size[0], self.size[1],
                               self.data_format) )
        return upsampled_input*mask
    def adjust_reference_tensor(self,ref_tensor,case='equal'):
        if case == 'equal':
            shape = K.int_shape(ref_tensor)
            rows, cols = self._get_rows_cols(shape)
            d_row = rows%self.strides[0]
            d_col = cols%self.strides[1]
            initial_row = 0
            initial_col = 0
            if d_row is not 0:
                initial_row = (rows-d_row)
            if d_col is not 0:
                initial_col = (cols-d_col)
            return self._adjust(ref_tensor,shape=(rows,cols))
    
    
    def _adjust (self,t,shape=None,**kwargs):
        if self.padding=='same':
            return self._adjust_same(t,shape)
    
    def _adjust_same(self,t,shape=None):
        min_val = K.min(t)
        rows, cols = shape
        r = rows%self.strides[0]
        c = cols%self.strides[1]
        pad_r, pad_c =0, 0
        
        if r!=0:
            pad_r = self.strides[0]-r
        if c!=0:
            pad_c = self.strides[1]-c
        if self.data_format=='channels_last':
            
            if pad_r*pad_c is not 0:
                u = min_val*K.ones_like(t[:,:,0:pad_c,:])
                t = K.concatenate([t,u],axis=2)
                v = min_val*K.ones_like(t[:,0:pad_r,:,:])
                t = K.concatenate([t,v],axis=1)
                
            elif pad_r == 0:
                u = min_val*K.ones_like(t[:,:,0:pad_c,:])
                t = K.concatenate([t,u],axis=2)
            else:
                v = min_val*K.ones_like(t[:,0:pad_r,:,:])
                t = K.concatenate([t,v],axis=1)
                
        if self.data_format=='channels_first':
            
            if pad_r*pad_c is not 0:
                u = min_val*K.ones_like(t[:,:,:,0:pad_c])
                t = K.concatenate([t,u],axis=2)
                v = min_val*K.ones_like(t[:,:,0:pad_r,:])
                t = K.concatenate([t,v],axis=1)
                
            elif pad_r == 0:
                u = min_val*K.ones_like(t[:,:,:,0:pad_c])
                t = K.concatenate([t,u],axis=2)
            else:
                v = min_val*K.ones_like(t[:,:,0:pad_r,:])
                t = K.concatenate([t,v],axis=1)
        return t
    
    
    def _adjust_valid(self,t,initial_row,initial_col,shape=None):
        min_val = K.min(t)
        rows, cols = shape
        if self.data_format=='channels_last':
            
            if initial_row*initial_col is not 0:
                u = min_val*K.ones_like(t[:,:,initial_col:cols,:])
                t = t[:,0:initial_row,0:initial_col,:]
                v = min_val*K.ones_like(t[:,initial_row:rows,0:initial_col,:])
                t = K.concatenate([t,v],axis=1)
                t = K.concatenate([t,u],axis=2)
            elif initial_row is 0:
                u = min_val*K.ones_like(t[:,:,initial_col:cols,:])
                t = t[:,:,0:initial_col,:]
                t = K.concatenate([t,u],axis=2)
            else:
                v = min_val*K.ones_like(t[:,initial_row:,:,:])
                t = t[:,0:initial_row,0:initial_col,:]
                t = K.concatenate([t,v],axis=1)
                
        if self.data_format=='channels_first':
            
            if initial_row*initial_col is not 0:
                u = min_val*K.ones_like(t[:,:,:,initial_col:])
                t = t[:,:,0:initial_row,0:initial_col]
                v = min_val*K.ones_like(t[:,:,initial_row:,:initial_col])
                t = K.concatenate([t,v],axis=2)
                t = K.concatenate([t,u],axis=3)
            elif initial_row is 0:
                u = min_val*K.ones_like(t[:,initial_col:,:,:])
                t = t[:,:,:,0:initial_col]
                t = K.concatenate([t,u],axis=3)
            else:
                v = min_val*K.ones_like(t[:,:,initial_row:,:])
                t = t[:,:,0:initial_row,0:initial_col]
                t = K.concatenate([t,v],axis=2)
        return t
                
            
    
    def call(self, inputs):
        output = self._unpooling_function(inputs=inputs,
                                        pool_size=self.pool_size,
                                        strides=self.strides,
                                        padding=self.padding,
                                        data_format=self.data_format)
        return output

    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'padding': self.padding,
                  'strides': self.strides,
                  'data_format': self.data_format}
        base_config = super((UnPooling2D), self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

if __name__ == '__main__':
    print('Nothing here to demonstrate.')