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
#        self.load_input_data(arg_dict['file_list'])
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
        self.load_input_data(arg_dict['file_list'])
        self.compute_predictions()
        self.compute_parts()
        self.compute_jaccard()
        self.compute_dice()
        print('Test Completed.\n\n')
#        self.view_maps()

    def set_model(self,model):
        self.model = None
        self.model = model
        
    def get_prediction_volume (self,model,test_vol_loc):
        self.clear_data()
        
    def clear_data(self):
        self.model = None
        self.X = None
        self.Y_pred = None
        self.Y_true = None
        self.dc_scores = None
        self.intersection = None
        self.abs_sum = None
        self.union = None
        self.jd_scores =None
        
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
    
#%%RISKY FUNCTIONS NEED TO #TODO_ CHANGE    
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
        
#%%Helpers
from workspace import generate_video_of_label_volume as gen_vid
#%%
class SingleVolumePredictor ():
    def __init__(self,batch_size=8,device='/gpu:1'):
        self.X = None
        self.Y_pred = None
        self.Y_true = None
        self.model = None
        self.num_ip = None#NUM of input branches in the network
        self.EXPECTED_SHAPE = (256,256,256,1)
        self.batch_size = batch_size
        self.shape = None
        self.device = device
        
    def get_prediction_on_batch(self,start_idx,end_idx):
        """X and model should be loaded before calling it. start_idx and \
            end_idx are inclusive."""
        if isinstance(self.X,list):#For the case of two inputs
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
        
    def load_input_data(self,file_loc,num_of_branches):
        self.num_ip = num_of_branches
        file = file_loc
        print('Loading T1/T2 from file:',file)
        with hf.File(file,'r') as f:
            T1=f['T1'][:]
            T2=f['T2'][:]
            
        self.X = [T1, T2]
        if self.num_ip == 1:
            self.X = np.concatenate([T1,T2],axis=CHANNEL_AXIS)
            self.shape = self.X.shape
            assert(self.X.shape == self.EXPECTED_SHAPE)
            print("X loaded: Shape:",self.shape)
        else:
            assert(self.X[0].shape == self.EXPECTED_SHAPE)
            self.shape = self.X[0].shape
            print("X[0] loaded: Shape:",self.X[0].shape)
        print('Num of inputs:',self.num_ip)
        
    def load_output_data(self,file):
        print('Loading true labels from:',file)
        with hf.File(TRAIN_FOLDER+'/'+file,'r') as f:
            self.Y_true=f['label'][:]
        print('Y_true_loaded:',self.Y_true.shape)
        f,q = np.unique(self.Y_true,return_counts=True)
        print("Unique Elems:",f,q)

    def get_label_volume(self,model,num_of_branches,test_file_loc=None,
                         show_predictions=False,window_prefix=''):
        """"if test_file_loc is not provided then input data loaded
            earlier will be used for prediction."""
        if test_file_loc is not None:
            print('Loading Input Data.')
            self.load_input_data(test_file_loc,num_of_branches)
        
        print("Updating SingleVolumePredictor's model:")
        self.model = model
        print(self.model,'with',num_of_branches,'branches.')
        with K.tf.device(self.device):
            self.compute_predictions()
        Y_predicted_volume = self.Y_pred
        if show_predictions:
            self.view_maps(window_prefix)
        self.clear_predictions()
        print('Y_predicted_volume:',Y_predicted_volume.dtype,Y_predicted_volume.shape)
        return Y_predicted_volume
    
    def clear_predictions (self):
        self.Y_pred = None
    
    def clear_volume_data (self):
        self.X = None
        self.Y_true = None
        print('Data cleared. Load another input volume before proceeding further.')
    
    def view_maps(self,window_prefix = ''):
        if self.Y_true is None:
            self.load_output_data(self.test_vol_dict)
        w = window_prefix
        while True:
            i= int(input('Enter Index'))
            if i == -1:
                break
            if isinstance(self.X,list):
                cv.imshow(w+"T1",self.X[0][i].astype('uint8'))
                cv.imshow(w+"T2",self.X[1][i].astype('uint8'))
            cv.imshow(w+"Y0_true",self.Y_true[i,:,:,0:1])
            cv.imshow(w+"Y1_true",self.Y_true[i,:,:,1:2])
            cv.imshow(w+"Y2_true",self.Y_true[i,:,:,2:3])
            cv.imshow(w+"Y3_true",self.Y_true[i,:,:,3:4])
            
            cv.imshow(w+"Y0",self.Y_pred[i,:,:,0:1])
            cv.imshow(w+"Y1",self.Y_pred[i,:,:,1:2])
            cv.imshow(w+"Y2",self.Y_pred[i,:,:,2:3])
            cv.imshow(w+"Y3",self.Y_pred[i,:,:,3:4])
            
            cv.imshow(w+"Y0_diff",np.abs(self.Y_pred[i,:,:,0:1]-self.Y_true[i,:,:,0:1]) )
            cv.imshow(w+"Y1intx",np.abs(self.Y_pred[i,:,:,1:2]-self.Y_true[i,:,:,1:2]) )
            cv.imshow(w+"Y2intx",np.abs(self.Y_pred[i,:,:,2:3]-self.Y_true[i,:,:,2:3]) )
            cv.imshow(w+"Y3intx",np.abs(self.Y_pred[i,:,:,3:4]-self.Y_true[i,:,:,3:4]) )
            cv.waitKey(0)
    
    def __del__ (self):
        del self.X
        del self.Y_true
        del self.Y_pred
        del self.model
    
    
import data as dt
class ModelEnsembler ():
    def __init__(self,load_model_fn):
        self.model_dict = None#{'0':['model1.h5','model2.h5']}
        self.test_vol_dict = None#{'0':/home/lessbg/test_vol1.h5, ...}
        self.Y_pred = ([],[],[])
        self.final_y_pred = None
        self.load_model = None
        self.set_load_model_function(load_model_fn)
        self.num_of_branches = None
        
    def set_load_model_function(self,load_model_fn):
        self.load_model = load_model_fn
    
    def ensemble_models_with_given_axis(self,axis,num_of_branches):
        model_list = self.model_dict[str(axis)]
        #list of the models belonging to given axis
        predictor = SingleVolumePredictor()
        #predictor oobject to apply predictors on given test volume.
        current_prediction = None
        for i in range (len(model_list)):
            print('Loading model from: ',model_list[i])
            if len(self.Y_pred[axis]) == 0:
                current_prediction = predictor.get_label_volume(self.load_model(model_list[i]),num_of_branches,
                                                       self.test_vol_dict[str(axis)])
                self.Y_pred[axis].append(current_prediction)
                print('First prediction along axis',axis,'computed.')
            else:
                current_prediction = predictor.get_label_volume(self.load_model(model_list[i]),2,
                                                       None)
                self.Y_pred[axis][0]= self.Y_pred[axis][0] + current_prediction
                print('Prediction number',i+1,' along axis',axis,'computed.')
#                predictor.clear_predictions()
            check_consistency(current_prediction,'current prediction')
            check_consistency(self.Y_pred[axis][0],'current self.Y_pred['+str(axis)+'][0]')
        print('Dividing axis',axis,'prediction by',len(model_list))        
        self.Y_pred[axis][0] = self.Y_pred[axis][0]/len(model_list)
        check_consistency(self.Y_pred[axis][0],'after division')
        print('Getting back original orientation:(from axis',axis,'->0)')
        self.Y_pred[axis][0] = dt.get_back_label_with_original_orientation(self.Y_pred[axis][0],axis)
        check_consistency(self.Y_pred[axis][0],'Original orientation')
        print('Deletin predictor object of axis',axis,'.')
        del predictor
            
    def combine_all_axis_predictions(self):
        num_of_axis_used= 0
        axis = 0
        for axis in range(3):
            if len(self.Y_pred[axis])==0:
                print('No predictions along axis',axis)
            else:
                num_of_axis_used+=1
                if self.final_y_pred is not None:
                    print('Adding to final pred')
                    self.final_y_pred = self.final_y_pred + self.Y_pred[axis][0]
                else:
                    print('Initializing final_pred')
                    self.final_y_pred = self.Y_pred[axis][0]
                check_consistency(self.final_y_pred,'current final_y_pred axis='+str(axis) )
        print('Num of axis used: ',num_of_axis_used)
        self.final_y_pred =self.final_y_pred/num_of_axis_used
        check_consistency(self.final_y_pred,' final_y_pred after division by'+str(num_of_axis_used))
        
    
    def compute_predictions_along_all_axis(self):
        for axis in range(3):
            if len(self.model_dict[str(axis)])>0:
                print('Initiating ensemble along axis',axis)
                self.ensemble_models_with_given_axis(axis,self.num_of_branches)
            else:
                print('No Model given for axis',axis)
        self.combine_all_axis_predictions()
    
    def reset(self):
        self.model_dict = None#{'0':['model1.h5','model2.h5']}
        self.test_vol_loc = None
        self.Y_pred = ([],[],[])
        self.final_y_pred = None
        
    def get_predicted_volume(self,model_dict,test_vol_dict,num_of_branches):
        self.reset()
        self.model_dict = model_dict
        self.test_vol_dict = test_vol_dict
        self.num_of_branches = num_of_branches
        self.compute_predictions_along_all_axis()
        self.Y_pred = ([],[],[])
        check_consistency(self.final_y_pred,' final_y_pred at return stage.')
        return self.final_y_pred
    
    def set_model_dict(self,model_dict):
        self.model_dict = model_dict
    
    def set_test_vol_dict(self,test_vol_dict):
        self.test_vol_dict = test_vol_dict
    
    def get_current_Y_pred(self,axis=None):
        if axis == None:
            return self.Y_pred
        else:
            return self.Y_pred[axis][0]
        
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
    
    def get_binary_map_after_thresholding (self,predicted_volume):
        assert(predicted_volume.shape==(256,256,256,4))
        for i in range(predicted_volume.shape[0]):
            predicted_volume[i] = self._get_4ch_binary_map(predicted_volume[i])
        check_consistency(predicted_volume,'Final volume after thresholding.')
        return predicted_volume
    
#%%CLASS

#%%SCORE_CALCULATOR
class ScoreCalculator:
    
    def __init__(self):
        self.intersection = None
        self.union = None
        self.abs_sum = None
        self.mask = None
        self.dc_scores = None
        self.jd_scores = None
        self.Y_true = None
        self.Y_pred = None
        self.epsilon = K.epsilon()
#%%SCORE COMPUTATION
    def compute_parts (self):
        self.intersection = np.abs(self.Y_true*self.Y_pred)
        print('Intersection:',self.intersection.shape)
        
        self.intersection = np.sum(self.intersection,axis=0)
        print('Intersection:',self.intersection.shape)
        
        self.intersection = np.sum(self.intersection,axis=0)
        print('Intersection:',self.intersection.shape)
        
        self.intersection = np.sum(self.intersection,axis=0)
        print('Intersection:',self.intersection.shape,self.intersection)
        
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
        
        self.abs_sum = np.sum(self.abs_sum,axis=0)
        print('Abs_sum:',self.abs_sum.shape,self.abs_sum)
        
        self.union = np.sum(self.union,axis=0)
        print('union:',self.union.shape)
        
        self.union = np.sum(self.union,axis=0)
        print('union:',self.union.shape)
        
        self.union = np.sum(self.union,axis=0)
        print('union:',self.union.shape,self.union)
#        print('INTERSECTION:',self.intersection)
#        print('UNION:',self.union)
        print('ABS_SUM:',self.abs_sum)
        
    def compute_jaccard(self):
        self.jd_scores = (self.intersection+self.epsilon)/(self.union+self.epsilon)
        print('Jaccard:',self.jd_scores.shape,self.jd_scores)

    def compute_dice(self):
        self.dc_scores = ((2.0*self.intersection)+self.epsilon)/(self.abs_sum+self.epsilon)
        print('Dice:',self.dc_scores.shape,self.dc_scores)
        
    def get_scores(self,Y_true,Y_pred):
        self.Y_pred = Y_pred
        if isinstance(Y_true,str):
            with hf.File(Y_true,'r') as f:
                self.Y_true = f['label'][:]
        else:
            self.Y_true = Y_true
#        self.Y_pred = Y_pred
        self.compute_parts()
        self.compute_jaccard()
        self.compute_dice()
        self.reset()
        return {'jaccard':self.jd_scores,'dice':self.dc_scores}
        
    def reset(self):
        self.intersection = None
        self.union = None
        self.abs_sum = None
        self.mask = None
        self.dc_scores = None
        self.jd_scores = None
        self.Y_true = None
        self.Y_pred = None
        
def main_run_test(model_dict,test_vol_dict):
    import train
    u_obj = train.util_obj
    ensembler = ModelEnsembler(u_obj.get_model)
    
    score_calculator = ScoreCalculator ()
    
    Y_pred = ensembler.get_predicted_volume(model_dict,test_vol_dict,2)
    Y_pred = ensembler.get_binary_map_after_thresholding(Y_pred)
    score_calculator.get_scores(test_vol_dict['0'],Y_pred)
    del ensembler
    del score_calculator
    del u_obj
    
def check_consistency(Y,msg):
    print(msg)
#    vid_name = input('Enter Video Name: ')
#    print('Generating Video:',vid_name)
#    gen_vid(Y,vid_name)
#    fq,uv = np.unique(Y,return_counts=True)
#    print('Max_val:',np.max(Y))
#    print('Min_val:',np.min(Y))
#    print('Mean: ',np.mean(Y))
#    print('Median: ',np.median(Y))
#    print('Sum:', np.sum(Y))
#    print('Std.Dev:',np.std(Y))
#    print('U_vals[1]',uv[1])
#    print('Fqs[1]', fq[1])
#    print('Video generated')
    


def testing_score_claculator():
    import numpy as np
    Y_true =  np.zeros(shape=(256,256,256,4),dtype='float32')
    Y_true[0,:,:,0]=1
    Y_true[1,:,:,1]=1
    Y_true[2,:,:,2]=1
    Y_true[3,0:200,0:200,3]=1
    np.set_printoptions(precision=3,threshold=np.inf)
#    print(Y_true)
    
    Y_pred = np.copy(Y_true)
    Y_pred[0,0:256,0:256,0]=0
    
    Y_pred[1,0:256,0:256,0]=1
    Y_pred[1,0:128,0:128,1]=0
    Y_pred[3,:,:,3] = 0
    Y_pred[3,0:192,0:192,3]=1
#    print(Y_true[3,:,:,3])
#    print('Y_pred',Y_pred[3,:,:,3])
    score_calc = ScoreCalculator()
    score_calc.get_scores(Y_true,Y_pred)    
    
if __name__ == '__main__':
    mf = MODEL_FOLDER
    trf = TRAIN_FOLDER
#%%
#    model_dict ={'0':[mf+'/Y3_axis0.h5',mf+'/Y5_axis0.h5'],
#                 '1':[mf+'/Y3_axis1.h5',mf+'/Y5_axis1.h5'],
#                 '2':[mf+'/Y3_axis2.h5',mf+'/Y5_axis2.h5']}
#    
#    test_vol_dict = {'0':trf+'/AXIS_0/subject-9-axis-0.h5',
#                     '1':trf+'/AXIS_1/subject-9-axis-1.h5',
#                     '2':trf+'/AXIS_2/subject-9-axis-2.h5'}
#    
#    with hf.File(trf+'/AXIS_0/subject-9-axis-0.h5','r') as f:
#        Y = f['label'][:]
#        check_consistency(Y,'TRUE LABELS')
#    main_run_test(model_dict,test_vol_dict)
##%%
#    model_dict ={'0':[mf+'/Y3_axis0.h5'],
#                 '1':[],
#                 '2':[]}
#    
#    test_vol_dict = {'0':trf+'/AXIS_0/subject-9-axis-0.h5',
#                     '1':trf+'/AXIS_1/subject-9-axis-1.h5',
#                     '2':trf+'/AXIS_2/subject-9-axis-2.h5'}
#    
#    main_run_test(model_dict,test_vol_dict)
##%%
#
#    model_dict ={'0':[],
#                 '1':[mf+'/Y3_axis1.h5'],
#                 '2':[]}
#    
#    test_vol_dict = {'0':trf+'/AXIS_0/subject-9-axis-0.h5',
#                     '1':trf+'/AXIS_1/subject-9-axis-1.h5',
#                     '2':trf+'/AXIS_2/subject-9-axis-2.h5'}
#    
#    main_run_test(model_dict,test_vol_dict)
##%%
#    model_dict ={'0':[],
#                 '1':[],
#                 '2':[mf+'/Y3_axis2.h5']}
#    
#    test_vol_dict = {'0':trf+'/AXIS_0/subject-9-axis-0.h5',
#                     '1':trf+'/AXIS_1/subject-9-axis-1.h5',
#                     '2':trf+'/AXIS_2/subject-9-axis-2.h5'}
#    
#    main_run_test(model_dict,test_vol_dict)
##%%
#    model_dict ={'0':[mf+'/Y3_axis0.h5'],
#                 '1':[mf+'/Y3_axis1.h5'],
#                 '2':[mf+'/Y3_axis2.h5']}
#    
#    test_vol_dict = {'0':trf+'/AXIS_0/subject-9-axis-0.h5',
#                     '1':trf+'/AXIS_1/subject-9-axis-1.h5',
#                     '2':trf+'/AXIS_2/subject-9-axis-2.h5'}
#    
#    main_run_test(model_dict,test_vol_dict)
#    


#%%
    model_dict ={'0':[mf+'/Y3_axis0.h5'],
                 '1':[],
                 '2':[]}
    
    test_vol_dict = {'0':trf+'/AXIS_0/subject-10-axis-0.h5',
                     '1':trf+'/AXIS_1/subject-10-axis-1.h5',
                     '2':trf+'/AXIS_2/subject-10-axis-2.h5'}
    
    main_run_test(model_dict,test_vol_dict)
#%%

    model_dict ={'0':[],
                 '1':[mf+'/Y3_axis1.h5'],
                 '2':[]}
    
    test_vol_dict = {'0':trf+'/AXIS_0/subject-10-axis-0.h5',
                     '1':trf+'/AXIS_1/subject-10-axis-1.h5',
                     '2':trf+'/AXIS_2/subject-10-axis-2.h5'}
    
    main_run_test(model_dict,test_vol_dict)
#%%
    model_dict ={'0':[],
                 '1':[],
                 '2':[mf+'/Y3_axis2.h5']}
    
    test_vol_dict = {'0':trf+'/AXIS_0/subject-10-axis-0.h5',
                     '1':trf+'/AXIS_1/subject-10-axis-1.h5',
                     '2':trf+'/AXIS_2/subject-10-axis-2.h5'}
    
    main_run_test(model_dict,test_vol_dict)
#%%
    model_dict ={'0':[mf+'/Y3_axis0.h5'],
                 '1':[mf+'/Y3_axis1.h5'],
                 '2':[mf+'/Y3_axis2.h5']}
    
    test_vol_dict = {'0':trf+'/AXIS_0/subject-10-axis-0.h5',
                     '1':trf+'/AXIS_1/subject-10-axis-1.h5',
                     '2':trf+'/AXIS_2/subject-10-axis-2.h5'}
    
    main_run_test(model_dict,test_vol_dict)