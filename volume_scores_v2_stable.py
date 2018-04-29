#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 12:40:23 2017

@author: shubham
"""

from default import *
import utility as utl
from keras import backend as K

class VolumePredictor:
    
    def __init__(self,model,arg_dict):
        '''file_list:[/axial/T1.h5, ...,]
        num_of_input_branches: 2
        
        '''
        self.args=arg_dict
        self.num_ip = arg_dict['num_of_input_branches']
        self.file_list = arg_dict['file_list']
        self.batch_size = arg_dict['batch_size']
        self.model = model
        self.test_dataset = None
        self.T1 = None
        self.T2 = None
        self.X = None
        self.Y_true  = None
        self.score_file = None
        self.Y_pred = None
        self.shape = None
        self.break_points = []
        #SCORE LOG OBJECTS
        self.jd_scores = []
        self.dc_scores = []
        
        self.abs_sum = None
        self.union = None
        self.intersection = None
        
        self.batch_abs_sum = None
        self.batch_union = None
        self.batch_intersection = None
        
        #OTHER THHINGS
        self.epsilon = K.epsilon()
        self.load_input_data(arg_dict['file_list'])
        self.u_obj = utl.Utility()

#%%loading input and output. Sync to save RAM
    def load_input_data(self,file_names):
        for i in range(len(file_names)):
            file = file_names[i]
            with hf.File(TRAIN_FOLDER+'/'+file,'r') as f:
                if i==0:
                    self.T1=f['T1'][:]
                    self.T2=f['T2'][:]
                    self.break_points.append(self.T1.shape[0])#begining of next
                    i+=1
                else:
                    self.T1=np.concatenate([self.T1,f['T1'][:]],axis=0)
                    self.T2=np.concatenate([self.T2,f['T2'][:]],axis=0)
                    self.break_points.append(self.T1.shape[0])
        self.X = [self.T1, self.T2]
        if self.num_ip == 1:
            self.X = np.concatenate([self.T1,self.T2],axis=3)
            del self.T1
            del self.T2
            self.shape = self.X.shape
            print("X loaded: Shape:",self.shape)
        else:
            self.shape = self.X[0].shape
            print("X loaded: Shape:",self.X[0].shape)
        print('Num of inputs:',self.num_ip)
            
    def load_output_data(self,file_names):
        for i in range(len(file_names)):
            file = file_names[i]
            with hf.File(TRAIN_FOLDER+'/'+file,'r') as f:
                if i==0:
                    self.Y_true=f['label'][:]
                    i+=1
                else:
                    self.Y_true=np.concatenate([self.Y_true,f['label'][:]],axis=0)
        print('Y_true_loaded:',self.Y_true.shape)
        f,q = np.unique(self.Y_true,return_counts=True)
        print("Unique Elems:",f,q)
#%%Getting Output
    def get_prediction_on_batch(self,start_idx,end_idx):
        if isinstance(self.X,list):
            x_batch = [self.X[0][start_idx:end_idx+1,:,:,:], self.X[1][start_idx:end_idx+1,:,:,:]]
        else:
            x_batch = self.X[start_idx:end_idx+1,:,:,:]
        
        return self.model.predict(x_batch)
    
    
    def compute_predictions(self):
        i_max = self.shape[0]//self.batch_size
        extra = self.shape[0]%self.batch_size
        for i in range(i_max):
            frm = i*self.batch_size
            to = frm+self.batch_size-1
            if i==0:
                self.Y_pred = self.get_prediction_on_batch(frm,to)
            else:
                self.Y_pred = np.concatenate([self.Y_pred,self.get_prediction_on_batch(frm,to)],axis=0)
        if extra>0:
            frm = i_max*self.batch_size
            to = frm + extra-1
            self.Y_pred = np.concatenate([self.Y_pred,self.get_prediction_on_batch(frm,to)],axis=0)
            
        
        print('Y_pred :',self.Y_pred.shape,self.Y_pred.dtype,type(self.Y_pred))
        for i in range(self.Y_pred.shape[0]):
            self.Y_pred[i] = self._get_4ch_binary_map(self.Y_pred[i])
        print('Y_pred :',self.Y_pred.shape,self.Y_pred.dtype,type(self.Y_pred))
        f,q = np.unique(self.Y_pred,return_counts=True)
        print("Unique Elems:",f,q)
#helpers to format the predictions    
    def _get_mask (self,nd_array):
        assert (nd_array.shape==(LENGTH,WIDTH,4))
        return np.argmax(nd_array,axis=2)
    
    
    def _get_binary_map (self,nd_array,mask,channel):
        assert (nd_array.shape==(LENGTH,WIDTH,4))
        y=np.zeros(shape=(LENGTH,WIDTH),dtype=nd_array.dtype)
        y[mask==channel]=1
        y = y.reshape(LENGTH,WIDTH,1)
        return y
    
    def _get_4ch_binary_map(self,nd_array):
        mask = self._get_mask(nd_array)
        y=[]
        for i in range(4):
            y.append(self._get_binary_map(nd_array,mask,channel=i))
        return np.concatenate(y,axis=2)
    
#%%SCORE COMPUTATION
    def compute_parts (self):
        self.load_output_data(self.file_list)
        self.intersection = np.abs(self.Y_true*self.Y_pred)
        print('Intersection:',self.intersection.shape)
        
        self.intersection = np.sum(self.intersection,axis=0)
        print('Intersection:',self.intersection.shape)
        
        self.intersection = np.sum(self.intersection,axis=0)
        print('Intersection:',self.intersection.shape)
        
        self.intersection = np.sum(self.intersection,axis=0)
        print('Intersection:',self.intersection.shape)
        
        self.abs_sum = np.abs(self.Y_true) + np.abs(self.Y_pred)
        print('Abs_sum:',self.abs_sum.shape)
        
        self.union = np.ones_like(self.abs_sum,dtype = self.abs_sum.dtype)
        print('Union:',self.union.shape)
        
        self.mask = np.less(self.abs_sum,self.union)
        print('mask:',self.mask.shape,self.mask.dtype)
        
        self.union [self.mask] = 0.
        print('union:',self.union.shape,self.union.dtype)
                   
        self.abs_sum = np.sum(self.abs_sum,axis=0)
        print('Abs_sum:',self.abs_sum.shape,self.abs_sum.dtype)
        
        self.abs_sum = np.sum(self.abs_sum,axis=0)
        print('Abs_sum:',self.abs_sum.shape,self.abs_sum.dtype)
        
        self.abs_sum = np.sum(self.abs_sum,axis=0) + self.epsilon
        print('Abs_sum:',self.abs_sum.shape)
        
        self.union = np.sum(self.union,axis=0)
        print('union:',self.union.shape)
        
        self.union = np.sum(self.union,axis=0)
        print('union:',self.union.shape)
        
        self.union = np.sum(self.union,axis=0) + self.epsilon
        print('union:',self.union.shape)
#        print('INTERSECTION:',self.intersection)
#        print('UNION:',self.union)
        print('ABS_SUM:',self.abs_sum)
        
    def compute_jaccard(self):
        self.jd_scores = self.intersection/self.union
        print('Jaccard:',self.jd_scores.shape,self.jd_scores)

    def compute_dice(self):
        self.dc_scores = 2.0*self.intersection/self.abs_sum
        print('Dice:',self.dc_scores.shape,self.dc_scores)
            
    def run_test(self):
        self.compute_predictions()
        self.compute_parts()
        self.compute_jaccard()
        self.compute_dice()
        print('Test Completed.\n\n')
#        self.view_maps()
        
    def view_maps(self):
        cv.namedWindow('T1')
        cv.namedWindow('T2')
        cv.namedWindow('Y0_true')
        cv.namedWindow('Y1_true')
        cv.namedWindow('Y2_true')
        cv.namedWindow('Y3_true')
        cv.namedWindow('Y0')
        cv.namedWindow('Y1')
        cv.namedWindow('Y2')
        cv.namedWindow('Y3')
        while True:
            i= int(input('Enter Index'))
            if i==-1:
                break
            if isinstance(self.X,list):
                cv.imshow("T1",self.X[0][i].astype('uint8'))
                cv.imshow("T2",self.X[1][i].astype('uint8'))
            cv.imshow("Y0_true",self.Y_true[i,:,:,0:1])
            cv.imshow("Y1_true",self.Y_true[i,:,:,1:2])
            cv.imshow("Y2_true",self.Y_true[i,:,:,2:3])
            cv.imshow("Y3_true",self.Y_true[i,:,:,3:4])
            
            cv.imshow("Y0",self.Y_pred[i,:,:,0:1])
            cv.imshow("Y1",self.Y_pred[i,:,:,1:2])
            cv.imshow("Y2",self.Y_pred[i,:,:,2:3])
            cv.imshow("Y3",self.Y_pred[i,:,:,3:4])
            
            cv.imshow("Y0intx",self.Y_pred[i,:,:,0:1]*self.Y_true[i,:,:,0:1])
            cv.imshow("Y1intx",self.Y_pred[i,:,:,1:2]*self.Y_true[i,:,:,1:2])
            cv.imshow("Y2intx",self.Y_pred[i,:,:,2:3]*self.Y_true[i,:,:,2:3])
            cv.imshow("Y3intx",self.Y_pred[i,:,:,3:4]*self.Y_true[i,:,:,3:4])
            cv.waitKey(0)
    
    
    def write_names_to_file(self,folder,file_name,names):
        with open(folder+"/"+file_name+"_dataset_names.dat",'w') as f:
            for items in names:
                f.write(items+"\n")
    
    
    def load_scores(self,score_file_loc):
        names_file_loc=(score_file_loc.split("scores.h5"))[0]+"_dataset_names.dat"
        names=[]
        with open(names_file_loc) as f:
            for items in f.readlines():
                names.append(items.strip())
        print(names)
        
        with hf.File(record_file_loc,'r') as f:
            record=f['history'][:]
        return (params,record)
    def __del__ (self):
        del self.X
        del self.Y_pred
        del self.Y_true
        del self.model
        del self.abs_sum
        del self.batch_abs_sum
        del self.mask
        del self.intersection
    
if __name__ == '__main__':
    import train
    u_obj = train.util_obj
    def show_scores(file,model):
        print('*'*5)
        files= [file]
        model_loc = MODEL_FOLDER + '/'+model
        print('Testing',model_loc,'on',files)
        model = u_obj.get_model(model_loc)
        args ={'num_of_input_branches':2,
               'file_list':files,
               'batch_size':1}
        vp = VolumePredictor(model,args)
        vp.run_test()
        del vp
        print('='*5)
        
    
    show_scores('subject-9-axis-0.h5','Y3_axis0.h5')
    show_scores('subject-10-axis-0.h5','Y3_axis0.h5')
    show_scores('subject-8-axis-0.h5','Y3_axis0.h5')
    
    show_scores('subject-9-axis-1.h5','Y3_axis1.h5')
    show_scores('subject-10-axis-1.h5','Y3_axis1.h5')
    show_scores('subject-8-axis-1.h5','Y3_axis1.h5')
    
    
    show_scores('subject-9-axis-2.h5','Y3_axis2.h5')
    show_scores('subject-10-axis-2.h5','Y3_axis2.h5')
    show_scores('subject-8-axis-2.h5','Y3_axis2.h5')