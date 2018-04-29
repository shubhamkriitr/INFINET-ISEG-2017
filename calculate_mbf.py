#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 17:06:42 2017

@author: abhijit
"""
from default import *
import numpy as np
l=['axial_with_less_bg_train.h5', 'coronal_with_less_bg_train.h5']


#with hf.File(TRAIN_FOLDER+"/"+l[i],'r') as f:
#    y=[]
#    for j in range(4):
#        x,q=np.unique(f['label'][:,:,:,j],return_counts=True)
#        y.append(q[1])
#        print('x',x,'y',y)

y=[]
for i in range(len(l)):
    with hf.File(TRAIN_FOLDER+"/"+l[i],'r') as f:
        print(l[i])
        for j in range(4):
            x,q=np.unique(f['label'][:,:,:,j],return_counts=True)
            print('File {} : Ch {} :'.format(i,j))
            print('X:',x,"\nQ:",q)
            if i>0:
                y[j]+=q[1]
            else:
                y.append(q[1])
            print('y[',j,']',y[j])
            print('y:',y)

y=np.array(y)
x=y
#print(y,np.sum(y,axis=1),np.sum(y,axis=0))
#
#x=y[:,1]
print(np.sum(x))
n=x/np.sum(x)
print(n)

w=np.median(n)/n
print('Weight',w)
print(w/np.sum(w))