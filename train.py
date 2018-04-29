#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 20:43:03 2017

@author: shubham
"""



#%%CHOOSE MODEL ARCHITECTURE
import MODEL_Y1 as net#Choose model architecture

#%%OTHER MODULES
from default import *
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger,ModelCheckpoint, EarlyStopping,Callback
from keras.preprocessing.image import (ImageDataGenerator, apply_transform,
                                       transform_matrix_offset_center,
                                       flip_axis)
import utility as utl

#Create net dependent Utility Object
# It will also be used during testing, as keras load_model() function expects the
# definition of custom functions used during compilation to be passed again
# during loading the saved model.
util_obj=utl.Utility({'load_model':net.load_keras_model})



#class to aid saving models at given interval.
class CheckPointManager(Callback):
    def __init__(self,model_folder,model_filename,interval=5):
        assert(model_filename[len(model_filename)-3:]=='.h5')
        self.file_loc=model_folder+"/"+model_filename[:-3]#
        self.interval=interval
        super(Callback, self).__init__()
    def on_epoch_end(self, epoch, logs={}):
        if((epoch+1)%self.interval==0):
            m_loc=self.file_loc+"_epoch"+str(epoch+1)+".h5"
            print("="*10,"Saving checkpoint:",m_loc,"="*10)
            self.model.save(m_loc)

#class to record history
class Logger(Callback):
    def __init__(self,log_folder,model_filename,log_interval=0):
        assert(model_filename[len(model_filename)-3:]=='.h5')
        self.file_loc=log_folder+"/"+model_filename[:-3]#
        self.folder=log_folder
        self.file_name=model_filename[:-3]
        self.epoch_log={}
        self.batch_log={}
        self.flag=0
        self.epoch=0#REMOVE_
        self.batch_count=0
        self.epoch_count=0
        super(Callback, self).__init__()
    def on_epoch_end(self, epoch, logs={}):
        self.epoch_count+=1
        if self.flag==1:
            for keys in logs:
                self.epoch_log[keys]=[logs[keys]]
            self.flag+=1
        else:
            for keys in logs:
                self.epoch_log[keys].append(logs[keys])
            name_e = self.file_name+"_epochwise_"+str(epoch)
            name_b = self.file_name+"_batchwise_"+str(epoch)
            util_obj.write_record_to_file(self.folder,name_e,util_obj.dict_to_array(self.epoch_log))
            util_obj.write_record_to_file(self.folder,name_b,util_obj.dict_to_array(self.batch_log))
        
        if self.epoch_count==15:
            print("="*100,'setting lr',"+"*100)
            import keras.backend as K
            self.model.optimizer.lr = K.variable(1e-2,K.floatx(),'lr')
        if self.epoch_count==30:
            print("="*100,'setting lr',"+"*100)
            import keras.backend as K
            self.model.optimizer.lr = K.variable(1e-3,K.floatx(),'lr')
        if self.epoch_count==45:
            print("="*100,'setting lr',"+"*100)
            import keras.backend as K
            self.model.optimizer.lr = K.variable(1e-3*0.5,K.floatx(),'lr')
        
    def on_batch_end(self,batch,logs={}):
        if self.flag==0:
            for keys in logs:
                self.batch_log[keys]=[logs[keys]]
            self.flag+=1
        else:
            for keys in logs:
                self.batch_log[keys].append(logs[keys])
        print("-"*8,"Batch end:",batch)
        print("logs:",logs)
        self.batch_count+=1
        if self.batch_count==100:
            print("="*100,'not setting lr',"+"*100)

    def on_train_end(self,logs={}):
        util_obj.write_record_to_file(self.folder,self.file_name+"_epochwise",util_obj.dict_to_array(self.epoch_log))
        util_obj.write_record_to_file(self.folder,self.file_name+"_batchwise",util_obj.dict_to_array(self.batch_log))

#load data
def load_data(dataset_file_names):
    for i in range( len(dataset_file_names)):
        f=hf.File(TRAIN_FOLDER+"/"+dataset_file_names[i],'r')
        if i==0:
            X1=f['T1'][:]
            X2=f['T2'][:]
            Y=f['label'][:]
        else:
            X1=np.concatenate([X1,f['T1'][:]],axis=0)
            X2=np.concatenate([X2,f['T2'][:]],axis=0)
            Y=np.concatenate([Y,f['label'][:]],axis=0)
        f.close()
    return (X1,X2,Y)

#function to format input depending on the network
def prepare_input(inputs,code=1):
    if code==1:#for single input with T1 and T2 as two channels
        X=np.concatenate(inputs,axis=CHANNEL_AXIS)
    elif code==2:#for two input branches one for T1 and other for T2
        X=inputs
    else:
        X=None
    return X

#function to execute training procedure
def train(names_dict,batch_size=10,epochs=1,interval=5,net_params={},
          begin_from_last_checkpoint=True):

    training_dataset_file_name=names_dict['training_dataset_file_names']
    validation_dataset_file_name=names_dict['validation_dataset_file_names']
    output_model_name=names_dict['output_model_file_name']

    X1,X2,Y= load_data(training_dataset_file_name)
    if validation_dataset_file_name==None:
        val_flag=False
    else:
        val_flag=True
        val_X1,val_X2,val_Y=load_data(validation_dataset_file_name)

    #handle dim ordering
    if IMAGE_DIM_ORDERING=='tf':
        pass
    elif IMAGE_DIM_ORDERING=='th':
        X1=X1.transpose(0,3,1,2)
        X2=X2.transpose(0,3,1,2)
        Y=Y.transpose(0,3,1,2)
        if val_flag:
            val_X1=val_X1.transpose(0,3,1,2)
            val_X2=val_X2.transpose(0,3,1,2)
            val_Y=val_Y.transpose(0,3,1,2)
    else:
        raise ValueError("Invalid value for IMAGE_DIM_ORDERING:",
                         IMAGE_DIM_ORDERING,"; it shoulde be 'tf' or 'th'.")

    #prepare input
    code=net_params['num_of_inputs']
    X=prepare_input([X1,X2],code)
    if val_flag:
        val_X=prepare_input([val_X1,val_X2],code)

    if begin_from_last_checkpoint:
        try:
            m=net.load_keras_model(MODEL_FOLDER+"/"+output_model_name)
            print("Last checkpoint loaded.")
        except IOError:
            print("There was no checkpoint. Training from scratch.")
            m=net.get_model((LENGTH,WIDTH,2),net_params)
    #callbacks
    logger=Logger(LOG_FOLDER, output_model_name)
    model_cp=CheckPointManager(MODEL_FOLDER, output_model_name,interval)
    csv_logger = CSVLogger(LOG_FOLDER+output_model_name[:-3]+"_logs.csv",
                           append=True, separator=';')

    #begin training
    print("#"*8,"About to train","#"*8)
    print("Input_shape:",X.shape)
    print("Output shape:",Y.shape)
    if val_flag:
        print("Val_Input_shape:",val_X.shape)
        print("Val_Output shape:",val_Y.shape)
    dummy=input('Press Enter to begin training.')
    if val_flag:
        m.fit(X,Y,batch_size=batch_size,epochs=epochs,validation_data=(val_X,val_Y),shuffle=True,
              callbacks=[model_cp,csv_logger,logger])
    else:
        m.fit(X,Y,batch_size=batch_size,epochs=epochs,shuffle=True,
              callbacks=[model_cp,csv_logger,logger])
    #save last checkpoint
    m.save(MODEL_FOLDER+"/"+output_model_name)



#%%Train with augmentation

##DEFAULT PARAMS
args_dict={'featurewise_center':False,
            'samplewise_center':False,
            'featurewise_std_normalization':False,
            'samplewise_std_normalization':False,
            'zca_whitening':False,
            'rotation_range':0.,
            'width_shift_range':0.,
            'height_shift_range':0.,
            'shear_range':0.,
            'zoom_range':0.,
            'channel_shift_range':0.,
            'fill_mode':'nearest',
            'cval':0.,
            'horizontal_flip':False,
            'vertical_flip':False,
            'rescale':None,
            'preprocessing_function':None,
            'data_format':None,
            'seed':0,
            'num_of_input_branches':1}
#to find shape of given input
def get_shape (list_of_files):
    shape=[]
    first=True
    for files in list_of_files:
        with hf.File(TRAIN_FOLDER+os.sep+files,'r') as f:
            if first:
                first=False
                shape=list(f['T1'].shape)
            else:
                assert(shape[1:]==list(f['T1'].shape[1:]))
                shape[0] = shape[0] + f['T1'].shape[0]
    return tuple(shape)

#function to execute training procedure
def train_with_gen(names_dict,batch_size=10,epochs=1,interval=5,net_params={},
          begin_from_last_checkpoint=True,generator_args=args_dict,validation_steps=None):
    #adding additional params to generator args
    generator_args['batch_size']=batch_size
    max_q_size= generator_args['max_q_size']
    
    training_dataset_file_name=names_dict['training_dataset_file_names']
    validation_dataset_file_name=names_dict['validation_dataset_file_names']
    output_model_name=names_dict['output_model_file_name']

    val_flag = validation_dataset_file_name is not None
    
    #calculating steps_per_epoch for training
    training_shape=get_shape(training_dataset_file_name)
    steps_per_epoch=int (training_shape[0]/batch_size) #TODO_ DONE:calculated
    print("Training data shape: ",training_shape)
    print('steps_per_epoch: ',steps_per_epoch)
    if val_flag:
        val_shape=get_shape(validation_dataset_file_name)
        if (validation_steps is None):
            validation_steps = int(val_shape[0]/batch_size)#TODO_ may add val_batch_size arg
        print('Validation Data Shape: ',val_shape)
        print('validation_steps: ', validation_steps)
        
       
    if begin_from_last_checkpoint:
        try:
            m=net.load_keras_model(MODEL_FOLDER+"/"+output_model_name)
            print("Last checkpoint loaded.")
        except IOError:
            print("There was no checkpoint. Training from scratch.")
            m=net.get_model((LENGTH,WIDTH,2),net_params)
    #callbacks
    logger=Logger(LOG_FOLDER, output_model_name)
    logger.set_model(m)
    model_cp=CheckPointManager(MODEL_FOLDER, output_model_name,interval)
    csv_logger = CSVLogger(LOG_FOLDER+output_model_name[:-3]+"_logs.csv",
                           append=True, separator=';')

    #Create data generators
    print('='*10,'Creating Generators','='*10)
    train_gen = DataGenerator(training_dataset_file_name,generator_args)
    if val_flag:
        val_gen = DataGenerator(validation_dataset_file_name,generator_args)
    
    #Begin Training
#    dummy=input('Press Enter to begin training.')
    if val_flag:
        m.fit_generator(train_gen,steps_per_epoch,epochs,validation_data=val_gen,
              callbacks=[model_cp,csv_logger,logger],validation_steps=validation_steps,
                        max_q_size=max_q_size)
    else:
        m.fit_generator(train_gen,steps_per_epoch,epochs,
              callbacks=[model_cp,csv_logger,logger],max_q_size=max_q_size)
    #save last checkpoint
    m.save(MODEL_FOLDER+"/"+output_model_name)


#%%EXTENSION OF IMAGEDATAGENERATOR OF KERAS



 #Params of Base ImageDataGenerator
#        featurewise_center=args['']
#        samplewise_center=args['']
#        featurewise_std_normalization=args['']
#        samplewise_std_normalization=args['']
#        zca_whitening=args['']
#        rotation_range=args['']
#        width_shift_range=args['']
#        height_shift_range=args['']
#        shear_range=args['']
#        zoom_range=args['']
#        channel_shift_range=args['']
#        fill_mode=args['']
#        cval=args['']
#        horizontal_flip=args['']
#        vertical_flip=args['']
#        rescale=args['']
#        preprocessing_function=args['']
#        data_format=args['']

class ImageGenerator(ImageDataGenerator):
    def __init__(self,args):
        ImageDataGenerator.__init__(self,
                 args['featurewise_center'],
                 args['samplewise_center'],
                 args['featurewise_std_normalization'],
                 args['samplewise_std_normalization'],
                 args['zca_whitening'],
                 args['rotation_range'],
                 args['width_shift_range'],
                 args['height_shift_range'],
                 args['shear_range'],
                 args['zoom_range'],
                 args['channel_shift_range'],
                 args['fill_mode'],
                 args['cval'],
                 args['horizontal_flip'],
                 args['vertical_flip'],
                 args['rescale' ],
                 args['preprocessing_function'],
                 args['data_format'])
       #Additional Params for this DataGenerator
        self.elastic_deformation=args['elastic_deformation']
    #An INDENT SHATTERED YOUR CONCEPTS OF CLASSES and took 3 hours of your life #TODO_ REMOVE IT
    def ed (self,img,mode='vertical',percentage_amp=0.8,num_of_ripples=10.0,shift_fn=None):
        if mode not in {'vertical','horizontal'}:
            raise ValueError('Mode should be either vertical or horizontal')
        if mode=='vertical':
            dim0=0
            dim1=1
        else:
            dim0=1
            dim1=0
        
        A= img.shape[dim0]*percentage_amp/100
        w= 2.0*np.pi*num_of_ripples/img.shape[dim1]
        
        if shift_fn is None:
            shift= lambda x: A*np.sin(w*x)
        else:
            shift=shift_fn
        
        if mode=='vertical':
            for i in range(img.shape[dim1]):
                img[:,i]=np.roll(img[:,i], int(shift(i)))
        else:
            for i in range(img.shape[dim1]):
                img[i,:]=np.roll(img[i,:], int(shift(i)))
    
        return img
    
    def random_transform(self, x):
#        print('New Transform Called=='*100)
        """Randomly augment a single image tensor.

        # Arguments
            x: 3D tensor, single image.

        # Returns
            A randomly transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0

        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                    [0, np.cos(shear), 0],
                                    [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
            x = apply_transform(x, transform_matrix, img_channel_axis,
                                fill_mode=self.fill_mode, cval=self.cval)

        if self.channel_shift_range != 0:
            x = random_channel_shift(x,
                                     self.channel_shift_range,
                                     img_channel_axis)
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_axis)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_axis)
        
        if self.elastic_deformation: #TODO_ remove True with some condition
            amp_percent= 0.5+0.4*np.random.random()
            num_ripples= int(5+6*np.random.random() )
            if np.random.random()<0.5:
                x= self.ed(x,'vertical',amp_percent,num_ripples)
            if np.random.random()<0.5:
                x= self.ed(x,'horizontal',amp_percent,num_ripples)

        return x
#%%Class for handling multiple inptuts/outputs/channels etc.
class DataGenerator:
    def __init__(self,file_names, args_dict):
        self.bs=args_dict['batch_size']
        self.num_inputs=args_dict['num_of_input_branches']
        self.img_generator=ImageGenerator(args_dict)
        self.seed=args_dict['seed']
        self.T1=None
        self.T2=None
        self.Y=None
        self.load_files(file_names)
        shape=(self.T1).shape
        
#        self.T1_gen=self.img_generator.flow(self.T1,batch_size=self.bs,
#                                              seed=self.seed)
#        self.T2_gen=self.img_generator.flow(self.T2,batch_size=self.bs,
#                                              seed=self.seed)
#        self.Y0_gen=self.img_generator.flow(self.Y[:,:,:,0].reshape(shape),batch_size=self.bs,
#                                              seed=self.seed)
#        self.Y1_gen=self.img_generator.flow(self.Y[:,:,:,1].reshape(shape),batch_size=self.bs,
#                                              seed=self.seed)
#        self.Y2_gen=self.img_generator.flow(self.Y[:,:,:,2].reshape(shape),batch_size=self.bs,
#                                              seed=self.seed)
#        self.Y3_gen=self.img_generator.flow(self.Y[:,:,:,3].reshape(shape),batch_size=self.bs,
#                                              seed=self.seed)
        #TODO_ Replace the snippet below by above one
        #TODO_ ALso check when shuffle=True
        #TODO_ Remove var batch_count
        self.batch_count=0
#        self.T1_gen=self.img_generator.flow(self.T1,batch_size=self.bs,
#                                              seed=self.seed,save_to_dir=OUTPUT_FOLDER+'/T1',shuffle=True)
#        self.T2_gen=self.img_generator.flow(self.T2,batch_size=self.bs,
#                                              seed=self.seed,save_to_dir=OUTPUT_FOLDER+'/T2',shuffle=True)
#        self.Y0_gen=self.img_generator.flow(self.Y[:,:,:,0].reshape(shape),batch_size=self.bs,
#                                              seed=self.seed,save_to_dir=OUTPUT_FOLDER+'/Y0',shuffle=True)
#        self.Y1_gen=self.img_generator.flow(self.Y[:,:,:,1].reshape(shape),batch_size=self.bs,
#                                              seed=self.seed,save_to_dir=OUTPUT_FOLDER+'/Y1',shuffle=True)
#        self.Y2_gen=self.img_generator.flow(self.Y[:,:,:,2].reshape(shape),batch_size=self.bs,
#                                              seed=self.seed,save_to_dir=OUTPUT_FOLDER+'/Y2',shuffle=True)
#        self.Y3_gen=self.img_generator.flow(self.Y[:,:,:,3].reshape(shape),batch_size=self.bs,
#                                              seed=self.seed,save_to_dir=OUTPUT_FOLDER+'/Y3',shuffle=True)
        self.T1_gen=self.img_generator.flow(self.T1,batch_size=self.bs,
                                              seed=self.seed,shuffle=True)
        self.T2_gen=self.img_generator.flow(self.T2,batch_size=self.bs,
                                              seed=self.seed,shuffle=True)
        self.Y0_gen=self.img_generator.flow(self.Y[:,:,:,0].reshape(shape),batch_size=self.bs,
                                              seed=self.seed,shuffle=True)
        self.Y1_gen=self.img_generator.flow(self.Y[:,:,:,1].reshape(shape),batch_size=self.bs,
                                              seed=self.seed,shuffle=True)
        self.Y2_gen=self.img_generator.flow(self.Y[:,:,:,2].reshape(shape),batch_size=self.bs,
                                              seed=self.seed,shuffle=True)
        self.Y3_gen=self.img_generator.flow(self.Y[:,:,:,3].reshape(shape),batch_size=self.bs,
                                              seed=self.seed,shuffle=True)

        
        
        
    def load_files(self,file_names):
        i=0
        for file in file_names:
            with hf.File(TRAIN_FOLDER+'/'+file,'r') as f:
                if i==0:
                    self.T1=f['T1'][:]
                    self.T2=f['T2'][:]
                    self.Y=f['label'][:]
                    i+=1
                else:
                    self.T1=np.concatenate([self.T1,f['T1'][:]],axis=0)
                    self.T2=np.concatenate([self.T2,f['T2'][:]],axis=0)
                    self.Y=np.concatenate([self.Y,f['label'][:]],axis=0)
    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in DataGenerator:
        return self
                        
    def __next__(self):
        #TODO_ Remove batch count and dummies
        self.batch_count+=1
        print("Generating Batch: ",self.batch_count)
#        dummy=input('Batch Number::'+str(self.batch_count))
        T1_batch=self.T1_gen.next()
#        dummy=input('got T1')
        T2_batch=self.T2_gen.next()
#        dummy=input('got T2')
        Y0_batch=self.Y0_gen.next()
#        dummy=input('got Y0')
        Y1_batch=self.Y1_gen.next()
#        dummy=input('got Y1')
        Y2_batch=self.Y2_gen.next()
#        dummy=input('got Y2')
        Y3_batch=self.Y3_gen.next()
#        dummy=input('got Y3')
        #ADDED_ remove later
#        print('Y0_batch:',Y0_batch.shape)
#        f,n=np.unique(Y0_batch,return_counts=True)
#        print(f)
#        print(n)
#        dummy=input('See values')
#        print('Y1_batch:',Y1_batch.shape)
#        f,n=np.unique(Y1_batch,return_counts=True)
#        print(f)
#        print(n)
#        dummy=input('See values')
#        print('Y2_batch:',Y2_batch.shape)
#        f,n=np.unique(Y2_batch,return_counts=True)
#        print(f)
#        print(n)
#        dummy=input('See values')
#        print('Y3_batch:',Y3_batch.shape)
#        f,n=np.unique(Y3_batch,return_counts=True)
#        print(f)
#        print(n)
#        dummy=input('See values')
        Y_batch= np.concatenate([Y0_batch,Y1_batch,Y2_batch,Y3_batch],axis=3)
        X_batch=[T1_batch,T2_batch]
        if self.num_inputs==1:
            X_batch=np.concatenate(X_batch,axis=3)
        elif self.num_inputs==2:
            pass
        return(X_batch,Y_batch)
        
  
