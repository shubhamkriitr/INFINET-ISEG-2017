#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 21:14:58 2017
Utility functions
@author: shubham
"""
from data import *
from keras.models import load_model
import time as tm

class Utility():
    def __init__(self,model_dependent_objects_dict={}):
        self.external_objects=model_dependent_objects_dict
        if 'load_model' in self.external_objects.keys():
            self.load_model=self.external_objects['load_model']
        print(self.external_objects)

    def echo(self,msg="Wait",duration=4):
        print('='*20,msg,'='*20)
        tm.sleep(duration)


    def getInfo(self,n,name="array"):
        print("-"*5,name,"-"*5)
        print(type(n),n.dtype,n.shape)
        print("---*******---")


    def view_4channel_array(self,nd_array,INTENSITY_VALUES=[0,0,0,255]):
        assert (nd_array.shape==(LENGTH,WIDTH,4))
        mask=np.argmax(nd_array,axis=2)
        y=np.zeros(shape=(LENGTH,WIDTH),dtype=np.uint8)
        for i in range(len(CHANNEL_VALUES)): 
            y[mask==i]=INTENSITY_VALUES[i]
        return y
    
    def get_model(self,model_loc):
        return (self.load_model)(model_loc)
        
    def prepare_input(self,inputs,code=1):
        if code==1:
            X=np.concatenate(inputs,axis=CHANNEL_AXIS)
        elif code==2:
            X=inputs
        else:
            X=None
        return X
    
    def get_model_output(self,model,dataset_loc,index,num_of_ips=1,normalize=False):
        with hf.File(dataset_loc,'r') as f:
            print(f,type(f))
            print(f['T1'])
            T1=f['T1']
            T2=f['T2']
            x1=get_sample(f,'T1',[index])
            x2=get_sample(f,'T2',[index])
            if normalize:
                print("Inputs are normalized.")
                x1=normalize_array(x1)
                x2=normalize_array(x2)
                print("Max values:",np.max(x1),np.max(x2))
            if num_of_ips==1:
                X=np.concatenate([x1.reshape(1,LENGTH,WIDTH,1),
                                  x2.reshape(1,LENGTH,WIDTH,1)],
                                    axis=3)
            elif num_of_ips==2:
                X=[x1,x2]
            else:
                raise ValueError('num_of_ips==',num_of_ips," is not allowed.")
            Y=model.predict(X)
            return (X,Y)
    
    def view_model_output(self,model_loc,dataset_loc,num_of_inputs=1,normalize=False,scale=1):
        print('Enter index= -1 to exit.')
        model=self.get_model(model_loc)
        f=hf.File(dataset_loc,'r')
        while True:
            index=int(input('Enter index:'))
            if index==-1:
                break
            X,Y=self.get_model_output(model,dataset_loc,index,num_of_inputs,normalize)
            if num_of_inputs==1:
                x1=X[:,:,:,0]*scale
                x2=X[:,:,:,1]*scale
            elif num_of_inputs==2:
                x1=X[0]*scale
                x2=X[1]*scale
            else:
                raise ValueError('')
            y=merge4ChannelArray(Y[0])
            y0=self.view_4channel_array(Y[0],[255,0,0,0])
            y1=self.view_4channel_array(Y[0],[0,255,0,0])
            y2=self.view_4channel_array(Y[0],[0,0,255,0])
            y3=self.view_4channel_array(Y[0],[0,0,0,255])
            
            cv.imshow("T1",x1[0].astype('uint8'))
            cv.imshow("T2",x2[0].astype('uint8'))
            cv.imshow("Predicted_y",y)
            cv.imshow("y0",y0)
            cv.imshow("y1",y1)
            cv.imshow("y2",y2)
            cv.imshow("y3",y3)
            cv.waitKey(0)
        f.close()

    
    def compare_model_output(self,model_loc,dataset_loc,num_of_inputs=1,normalize=False,scale=1):
        print('Enter index= -1 to exit.')
        model=self.get_model(model_loc)
        f=hf.File(dataset_loc,'r')
        while True:
            index=int(input('Enter index:'))
            if index==-1:
                break
            X,Y=self.get_model_output(model,dataset_loc,index,num_of_inputs,normalize)
            Yo=f['label'][index]
            if num_of_inputs==1:
                x1=X[:,:,:,0]*scale
                x2=X[:,:,:,1]*scale
            elif num_of_inputs==2:
                x1=X[0]*scale
                x2=X[1]*scale
            else:
                raise ValueError('')
            y=merge4ChannelArray(Y[0])
            y0=self.view_4channel_array(Y[0],[255,0,0,0])
            y1=self.view_4channel_array(Y[0],[0,255,0,0])
            y2=self.view_4channel_array(Y[0],[0,0,255,0])
            y3=self.view_4channel_array(Y[0],[0,0,0,255])
            
            y_0=self.view_4channel_array(Yo,[255,0,0,0])
            y_1=self.view_4channel_array(Yo,[0,255,0,0])
            y_2=self.view_4channel_array(Yo,[0,0,255,0])
            y_3=self.view_4channel_array(Yo,[0,0,0,255])
            cv.imshow("T1",x1[0].astype('uint8'))
            cv.imshow("T2",x2[0].astype('uint8'))
            cv.imshow("Predicted_y",y)
            cv.imshow("y0",y0)
            cv.imshow("y1",y1)
            cv.imshow("y2",y2)
            cv.imshow("y3",y3)
            
            cv.imshow("Actual_Y",Yo)
            cv.imshow("Actual_y0",y_0)
            cv.imshow("Actual_y1",y_1)
            cv.imshow("Actual_y2",y_2)
            cv.imshow("Actual_y3",y_3)
            cv.waitKey(0)
        f.close()

#for testing the performance on validation dataset
    def evaluate_model(self,model_loc,dataset_loc,batch_size=32,num_of_inputs=1):
        model=self.get_model(model_loc)
        f=hf.File(dataset_loc,'r')
        X1=f['T1'][:]
        X2=f['T2'][:]
        Y=f['label'][:]
        #order [X1,X2] is important
        code=num_of_inputs
        X=self.prepare_input([X1,X2],code)
        model.evaluate(X,Y,batch_size)
        
    #for maintaing and viewing logs
    def dict_to_array(self,logs):
        flag=True
        params=list(logs.keys())
        num_of_entries=len(logs[params[0]])
        len_=[]
        for keys in logs:
            len_.append((keys,len(logs[keys])))
            if num_of_entries is not len(logs[keys]):
                flag=False
        if not flag:
            print("-"*10,"Log Params do not have equal number of records.\n",len_,"-"*10)
        
        record=np.zeros(shape=(len(params),num_of_entries),dtype='float32')
        print("params:",params)
        for i in range(record.shape[0]):
            record[i]=logs[params[i]]
        print(record)
        return (params,record) 
    
    
    def write_record_to_file(self,folder,file_name,params_and_record=()):
        with open(folder+"/"+file_name+"_params.dat",'w') as f:
            for items in params_and_record[0]:
                f.write(items+"\n")
            
        with hf.File(folder+"/"+file_name+"_record.h5",'w') as f:
            f.create_dataset("history",data=params_and_record[1])
    
    
    def load_history(self,record_file_loc):
        params_file_loc=(record_file_loc.split("record.h5"))[0]+"params.dat"
        params=[]
        with open(params_file_loc) as f:
            for items in f.readlines():
                params.append(items.strip())
        print(params)
        
        with hf.File(record_file_loc,'r') as f:
            record=f['history'][:]
        return (params,record)
    
    
    def plot_history (self,params_and_record=(),x_label="epochs"):
        import matplotlib.pyplot as plt
        params,record=params_and_record
        x=np.arange(record.shape[1])
        current_row=1
        for i in range (len(params)):
            fig=plt.figure(i)
            plt.title('History')
            plt.xlabel(x_label)
            plt.ylabel(params[i])
            plt.plot(x,record[i])
            current_row+=1
        plt.show()
        
        
        
        
if __name__=='__main__':
    utl=Utility()
    utl.plot_history(utl.load_history(LOG_FOLDER+"/unet_with_sgd_a1_m1_batchwise_record.h5"),'batch')
#    utl.view_model_output(MODEL_FOLDER+os.sep+"unet_model7SGD.h5",
#                          TRAIN_FOLDER+os.sep+"train_iseg_1.h5")
#    
    
