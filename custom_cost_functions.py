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
np.set_printoptions(threshold=np.inf)
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
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection=0
    #med_bal_factor=[210/16000,210/176,210/378,210/246]
#    med_bal_factor=[0.04995437, 1.73594757, 0.82916348, 1.25950102]
    med_bal_factor = [1,1,1,1]#TODO_ remove it. After testing
    union=0
    for i in range(4):
        intersection+= med_bal_factor[i]*(K.sum(y_true[:,:,:,i] * y_pred[:,:,:,i]))
        union+=med_bal_factor[i]*(K.sum(y_true[:,:,:,i]+y_pred[:,:,:,i]))
    return (2. * intersection + smooth) / (union + smooth)

#%%CLASS-WISE-DICE
def dice_coef_axis(y_true, y_pred,i):
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection=0

    med_bal_factor = [1,1,1,1]#TODO_ remove it. After testing
    union=0
    intersection+= med_bal_factor[i]*(K.sum(y_true[:,:,:,i] * y_pred[:,:,:,i]))
    union+=med_bal_factor[i]*(K.sum(y_true[:,:,:,i]+y_pred[:,:,:,i]))
    return (2. * intersection + smooth) / (union + smooth)

def dice_coef_0(y_true,y_pred):
    return dice_coef_axis(y_true,y_pred,0)
def dice_coef_1(y_true,y_pred):
    return dice_coef_axis(y_true,y_pred,1)
def dice_coef_2(y_true,y_pred):
    return dice_coef_axis(y_true,y_pred,2)
def dice_coef_3(y_true,y_pred):
    return dice_coef_axis(y_true,y_pred,3)

#%%

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
#%%2.MIXED_LOSS (MIXED DICE LOSS AND BINARY CROSSENTROPY LOSS) 
#################
#MIXED_LOSS_TEST (MIXED DICE LOSS AND BINARY CROSSENTROPY LOSS) 
def old_mixed_loss (y_true,y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
    med_bal_factor=[0.04125156, 1.73594757, 0.82916348, 1.25950102]#TODO_ remove
    edge_weights = get_grad_tensor(y_true, False)
    return edge_weights
    tensor_list=[]
    for i in range(len(med_bal_factor)):
        tensor_list.append(med_bal_factor[i]*K.ones_like(edge_weights[:,:,:,i:i+1]) )
    class_weights = K.concatenate(tensor_list, axis=3)
    final_weights = K.tf.add(edge_weights, class_weights)
    roi = 1 - y_true[:,:,:,0:1]
    roi_list=[]
    for i in range(4):
        roi_list.append(roi )
    roi_weight = K.concatenate(roi_list, axis=3)
    roi_weight = 5.0*K.clip(roi_weight,K.epsilon(), 1 - K.epsilon())
    final_weights = K.tf.add(final_weights, roi_weight)
    return final_weights
    cross_entropy_part = - (K.tf.reduce_mean(K.tf.multiply(y_true * K.tf.log(y_pred),
                                             final_weights)))
    weighted_intersection = K.sum(y_true*y_pred*final_weights)
    union = K.sum(y_true) + K.sum(y_pred)
    dice_part = -2.0*(weighted_intersection/union)
    return (cross_entropy_part + dice_part)#TODO_ remove modifications
    
#%%2.MIXED_LOSS (MIXED DICE LOSS AND BINARY CROSSENTROPY LOSS) 
#################
#MIXED_LOSS_TEST (MIXED DICE LOSS AND BINARY CROSSENTROPY LOSS) 
#THE CHANNEL WISE CONTRIBUTION TO THE LOSS HAS BEEN CALCULATED SEPERATELY AND
#SUMMED UP, TO SEE IF THERE IS ANY EFFECT OF DIRECT CHANNELIZATION OF 
#GRRADIENTS FROM THE CHANNEL WITH MORE ERROR
def mixed_loss (y_true,y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
    med_bal_factor=[0.01942449 , 1.73594757 , 0.82916348 , 1.25950102]#[0.04125156, 1.73594757, 0.82916348, 1.25950102]#TODO_ remove
    edge_weights = 2*max(med_bal_factor)*get_grad_tensor(y_true, False)
#    return edge_weights
    tensor_list=[]
    for i in range(len(med_bal_factor)):
        tensor_list.append(med_bal_factor[i]*K.ones_like(edge_weights[:,:,:,i:i+1]) )
    class_weights = K.concatenate(tensor_list, axis=3)
    final_weights = K.tf.add(edge_weights, class_weights)
#    return final_weights
    cross_entropy_part = 0.0
    for i in range(4):
        cross_entropy_part += -1.0*(K.tf.reduce_mean(K.tf.multiply(y_true[:,:,:,i] * K.tf.log(y_pred[:,:,:,i]),
                                                 final_weights[:,:,:,i])))
    dice_part = 0.0
    for i in range(4):
        weighted_intersection = K.sum(y_true[:,:,:,i]*y_pred[:,:,:,i]*final_weights[:,:,:,i])
        union = K.sum(y_true[:,:,:,i]) + K.sum(y_pred[:,:,:,i])
        dice_part += -2.0*(weighted_intersection/union)
    return (cross_entropy_part + dice_part)#TODO_ remove modifications

#%%3.EDGE WEIGHTED LOSS
def stable_get_edge_kernel():
    s=np.array([[1,  2,  -1],
                [2,  0, -2],
                [1, -2, -1]], dtype='float32')
    ker=np.zeros(shape=(3,3,4,4),dtype='float32')
    for i in range(4):
        ker[:,:,i,i]=s
    ker=K.constant(ker)
    return ker

def get_edge_kernel():
    s=np.array([[1,  2,  -1],
                [2,  0, -2],
                [1, -2, -1]], dtype='float32')
    ker=np.zeros(shape=(3,3,4,4),dtype='float32')
    ker[:,:,0,0]=s
    ker[:,:,1,1]=s
    ker[:,:,2,2]=s
    ker[:,:,3,3]=s
       
    ker=K.constant(ker)
    return ker

print('='*100,'New Loss')
KERNEL=get_edge_kernel()

#%%
def get_gauss_kernel():
    s=np.array([[16,  26,  16],
                [26,  41, 26],
                [16, 26, 16]], dtype='float32')
    s= s/np.sum(s)
    ker=np.zeros(shape=(3,3,4,4),dtype='float32')
    ker[:,:,0,0]=s
    ker[:,:,1,1]=s
    ker[:,:,2,2]=s
    ker[:,:,3,3]=s
       
    ker=K.constant(ker)
    return ker

GAUSS_KERNEL = get_gauss_kernel()
def smooth_edge_tensor (edge_tensor):
    return K.conv2d(edge_tensor,GAUSS_KERNEL,padding='same')

#%%GET EDGE TENSOR

def get_primary_edge_tensor(img):
    return K.abs(K.conv2d(img,KERNEL,padding='same'))

def get_edge_tensor(img):
    edge_tensor = get_primary_edge_tensor(img)
    bias_for_background=0.#TODO_ keep it < 1 if all edges are 1
    value_for_edges=5.0
                             
    edges =value_for_edges*K.sum(edge_tensor,axis=3,keepdims=True)
#    edges =1*K.sum(edge_tensor,axis=3,keepdims=True)
#    bias =( K.ones_like(edges,dtype=edges.dtype) )*bias_for_background
#    edges = K.tf.add(edges,bias)
    
    tensor_list=[]
    class_weights=[1, 1, 1, 1]#TODO_ remove the loop below & class_weights
    for i in range(len(class_weights)):
        tensor_list.append(class_weights[i]*edges)
    
    img_tensor = K.concatenate(tensor_list, axis=3)
    img_tensor = K.clip(img_tensor,bias_for_background,value_for_edges)
    return img_tensor#/K.max(img_tensor)
    
#%%NEW EDGE TENSOR
def get_sobel_kernel(axis):
    s=np.array([[1,  2,  1],
                [0,  0, 0],
                [-1, -2, -1]], dtype='float32')
    if axis=='y':
        pass
    elif axis == 'x':
        s = np.transpose(s,)
    ker=np.zeros(shape=(3,3,4,4),dtype='float32')
    ker[:,:,0,0]=s
    ker[:,:,1,1]=s
    ker[:,:,2,2]=s
    ker[:,:,3,3]=s
       
    ker=K.constant(ker)
    return ker

SOBEL_X = get_sobel_kernel('x')
SOBEL_Y = get_sobel_kernel('y')

def  get_grad_tensor (img_tensor, apply_gauss = True):
    grad_x = K.conv2d(img_tensor,SOBEL_X,padding='same')
    grad_y = K.conv2d(img_tensor,SOBEL_Y,padding='same')
    grad_tensor = K.sqrt(grad_x*grad_y + grad_y*grad_y)
    
    grad_tensor = K.greater(grad_tensor, 100.0*K.epsilon())
    grad_tensor = K.cast(grad_tensor, K.floatx())
    grad_tensor = K.clip(grad_tensor, K.epsilon(),1.0)
    grad_map = K.sum(grad_tensor,axis=CHANNEL_AXIS,keepdims= True)
    grad_map = [grad_map, grad_map]
    grad_tensor = K.concatenate(grad_map, axis = CHANNEL_AXIS)
    del grad_map
    grad_tensor = K.concatenate([grad_tensor, grad_tensor], axis = CHANNEL_AXIS)
    grad_tensor = K.greater(grad_tensor, 100.0*K.epsilon())
    grad_tensor = K.cast(grad_tensor, K.floatx())
    print("K.floatx",K.floatx())
    if apply_gauss:
        grad_tensor = K.conv2d(grad_tensor,GAUSS_KERNEL,padding='same')
    return grad_tensor
#%%EDGE LOSS
def edge_loss(y_true,y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
    med_bal_factor=[0.04995437, 1.73594757, 0.82916348, 1.25950102]
    class_weights= med_bal_factor
    tensor_list=[]
    for i in range(len(class_weights)):
        tensor_list.append(class_weights[i]*y_true[:,:,:,i:i+1])
    y_true_weighted = K.concatenate(tensor_list, axis=3)
    return - (K.tf.reduce_mean(K.tf.multiply(y_true_weighted * K.tf.log(y_pred),
                                             get_edge_tensor(y_true))))
def edge_loss_for_con_check(y_true,y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
    class_weights=[1, 1, 2, 2]
    tensor_list=[]
    for i in range(len(class_weights)):
        tensor_list.append(class_weights[i]*y_true[:,:,:,i:i+1])
    y_true_weighted = K.concatenate(tensor_list, axis=3)
    return y_true_weighted
    return - (K.tf.reduce_mean(K.tf.multiply(y_true_weighted * K.tf.log(y_pred),
                                             get_edge_tensor(y_true))))

def stable_edge_loss(y_true,y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1. - K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
    return - (K.tf.reduce_mean(K.tf.multiply(y_true * K.tf.log(y_pred),
                                             get_edge_tensor(y_true))))
    
#%%Testing module's functions
if __name__=='__main__':
    print(K.get_session())
    print('KERNEL:\n',KERNEL,"\n",KERNEL.eval(session=K.get_session()))
    print(K.get_session())
    mod_name='model_b2_n7_lr3_bs20_mixed.h5'#'unet_with_sgd_edge_loss.h5'
    file_loc=TRAIN_FOLDER+os.sep+'axial_with_less_bg.h5'
#    file_loc=dlt.TEST_FOLDER+dlt.os.sep+'test_set_coronal.h5'
    #'model_b2_n6_lr3_bs20_dice_y3_weighted.h5'#'model_b1_n1.h5'
    
#%%
    def test_mixed_loss_fn(file_loc):
        with hf.File(file_loc,'r') as f:
            i=128
            while i>=0:
                i=int(input('Enter Index:'))
                num=int(input('Num of samples'))
                x1=f['T1'][i:i+num,:,:,0:1]
                x2=f['T2'][i:i+num,:,:,0:1]
                
                Y=f['label'][i:i+num,:,:,:]
                Y=K.tf.convert_to_tensor(Y,dtype=Y.dtype)
                Z= (mixed_loss(Y,Y)).eval(session=K.get_session())
#                print('Edit in mixed_loss_for_test')
#                print('Loss:',Z)
#                return
                for s in range(num):
                    y = []
                    for i in range(4):
                        y.append(Z[s,:,:,i:i+1])
                    
                                  
                    for i in range(len(y)):
                        print("Channel:",i)
                        print(np.unique(y[i], return_counts=True),(y[i]).shape)
                        y_show=(y[i])/np.max(np.array(y) )
                        y_show=y_show*255
                        print(np.unique(y[i]))
                        cv.imshow('Edges_'+str(i),y_show.astype('uint8'))
                    cv.waitKey(0)
    
    test_mixed_loss_fn(file_loc)
    def view_4channel_array(nd_array,INTENSITY_VALUES=[0,0,0,255]):
        assert (nd_array.shape==(LENGTH,WIDTH,4))
        mask=np.argmax(nd_array,axis=2)
        y=np.zeros(shape=(LENGTH,WIDTH),dtype=np.uint8)
        for i in range(len(CHANNEL_VALUES)): 
            y[mask==i]=INTENSITY_VALUES[i]
        return y

#%%
    def test_edge_loss_fn(file_loc):
        
        with hf.File(file_loc,'r') as f:
            i=128
            while i>=0:
                i=int(input('Enter Index:'))
                num=int(input('Num of samples'))
                x1=f['T1'][i:i+num,:,:,0:1]
                x2=f['T2'][i:i+num,:,:,0:1]
                
                Y=f['label'][i:i+num,:,:,:]
                Y=K.tf.convert_to_tensor(Y,dtype=Y.dtype)
                Z= (edge_loss_for_con_check(Y,Y)).eval(session=K.get_session())
                for s in range(num):
                    y = []
                    for i in range(4):
                        y.append(Z[s,:,:,i:i+1])
                    
                                  
                    for i in range(len(y)):
                        print("Channel:",i)
                        print(np.unique(y[i]),(y[i]).shape)
                        y_show=(y[i])/np.max(np.array(y) )
                        y_show=y_show*255
                        print(np.unique(y[i]))
                        cv.imshow('Edges_'+str(i),y_show.astype('uint8'))
                    cv.waitKey(0)
    
#    test_edge_loss_fn(file_loc)

#%%

    def test_edge_tensor_function(file_loc):
        with hf.File(file_loc,'r') as f:
            i=128
            while i>=0:
                i=int(input('Enter Index:'))
                num=int(input('Num of samples'))
                x1=f['T1'][i:i+num,:,:,0:1]
                x2=f['T2'][i:i+num,:,:,0:1]
                
                Y=f['label'][i:i+num,:,:,:]
                Z= (get_edge_tensor(Y)).eval(session=K.get_session())
                print('Z_tensor:',Z.shape)
                fq,val=np.unique(Z,return_counts=True)
                print(fq,val)
                for s in range(num):
                    y0=view_4channel_array(Y[s],[255,0,0,0])
                    y1=view_4channel_array(Y[s],[0,255,0,0])
                    y2=view_4channel_array(Y[s],[0,0,255,0])
                    y3=view_4channel_array(Y[s],[0,0,0,255])
                    
                    y = []
                    for i in range(4):
                        y.append(Z[s,:,:,i:i+1])
                    
                                  
                    for i in range(len(y)):
                        print("Channel:",i)
                        print(np.unique(y[i]),(y[i]).shape)
                        y_show=(y[i])/np.max(np.array(y) )
                        y_show=y_show*255
                        print(np.unique(y[i]))
                        cv.imshow('Edges_'+str(i),y_show.astype('uint8'))
    #                if i==2:
    #                    u=np.unique(y[i])
    #                    for j in range(len(u)):
    #                        mask=(y[i]==u[j])
    #                        img=np.zeros_like(y[i])
    #                        img[mask]=255
    #                        cv.imshow("val"+str(u[j]),img.astype('uint8'))
                            
                
                    cv.imshow('Actual_yo',y0)
                    cv.imshow('Actual_y1',y1)
                    cv.imshow('Actual_y2',y2)
                    cv.imshow('Actual_y3',y3)
                    cv.waitKey(0)