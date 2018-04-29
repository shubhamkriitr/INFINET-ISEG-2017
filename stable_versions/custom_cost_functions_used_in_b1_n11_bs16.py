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
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection=0
    #med_bal_factor=[210/16000,210/176,210/378,210/246]
    med_bal_factor=[0.0195,1.7747,0.8253,1.2683]
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
def stable_get_edge_tensor(img):
    return K.abs(K.conv2d(img,KERNEL,padding='same'))

def get_primary_edge_tensor(img):
    return K.abs(K.conv2d(img,KERNEL,padding='same'))

def older_get_edge_tensor(img):
    img_tensor = get_primary_edge_tensor(img)
    arr = img_tensor.eval(session=K.get_session())
    aux=np.ones_like(arr,dtype=arr.dtype)
    bias_for_background=np.max(arr)/10
#    print('AUX::::',aux.shape)
    for i in range(4):
        aux[:,:,:,i]= np.sum(arr,axis=3)+bias_for_background
    aux[:,:,:,3]=aux[:,:,:,3]*10
    aux[:,:,:,2]=aux[:,:,:,2]*10
    aux[:,:,:,1]=aux[:,:,:,1]*5
    
    img_tensor = K.tf.convert_to_tensor(aux)
#    img_tensor=K.tf.add(img_tensor,base)
    return img_tensor/K.max(img_tensor)

def get_edge_tensor(img):
    edge_tensor = get_primary_edge_tensor(img)
    bias_for_background=K.max(edge_tensor)/10
    edges = K.sum(edge_tensor,axis=3,keepdims=True)
    
    tensor_list=[]
    class_weights=[1, 5, 20, 10]
    for i in range(len(class_weights)):
        tensor_list.append(class_weights[i]*edges)
    
    img_tensor = K.concatenate(tensor_list, axis=3)
    return img_tensor/K.max(img_tensor)

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
    mod_name='model_b2_n7_lr3_bs20_mixed.h5'#'unet_with_sgd_edge_loss.h5'
    #'model_b2_n6_lr3_bs20_dice_y3_weighted.h5'#'model_b1_n1.h5'
    def view_4channel_array(nd_array,INTENSITY_VALUES=[0,0,0,255]):
        assert (nd_array.shape==(LENGTH,WIDTH,4))
        mask=np.argmax(nd_array,axis=2)
        y=np.zeros(shape=(LENGTH,WIDTH),dtype=np.uint8)
        for i in range(len(CHANNEL_VALUES)): 
            y[mask==i]=INTENSITY_VALUES[i]
        return y
    file_loc=TRAIN_FOLDER+os.sep+'axial_with_less_bg_train.h5'
#    file_loc=dlt.TEST_FOLDER+dlt.os.sep+'test_set_coronal.h5'
    with hf.File(file_loc,'r') as f:
        i=128
        while i>=0:
            i=int(input('Enter Index:'))
            x1=f['T1'][i,:,:,0:1]
            x2=f['T2'][i,:,:,0:1]
            cv.imshow('input_T1',x1.astype('uint8'))
            cv.imshow('input_T2',x2.astype('uint8'))
            Y=f['label'][i:i+1,:,:,:]
            y0=view_4channel_array(Y[0],[255,0,0,0])
            y1=view_4channel_array(Y[0],[0,255,0,0])
            y2=view_4channel_array(Y[0],[0,0,255,0])
            y3=view_4channel_array(Y[0],[0,0,0,255])
            Y= (get_edge_tensor(Y)).eval(session=K.get_session())
            print('Y_tensor:',Y.shape)
            


            fq,val=np.unique(Y,return_counts=True)
            print(fq,val)
            y = []
            for i in range(4):
                y.append(Y[0,:,:,i:i+1])
            y1p3= y[2]+y[3]
            y.append(y1p3)
            for i in range(len(y)):
                print("Channel:",i)
                print(np.unique(y[i]),(y[i]).shape)
                y[i]=(y[i])/np.max(y[i])
                y[i]=y[i]*255
                print(np.unique(y[i]))
                cv.imshow('Edges_'+str(i),y[i].astype('uint8'))
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