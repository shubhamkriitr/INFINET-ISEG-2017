	#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 19:01:18 2017
@name:data.py
@description:module to handle dataset formation, processing and retrieval
@author: Shubham Kumar
"""
from default import *
import warnings


def readFileAsNDArray (img_loc):
    """reads hdr/img file and outputs corresponding numpy array. 
    The function is case specific(CSF)."""
    data=nib.load(img_loc)
    image_data=data.get_data()
    if (image_data.shape!=(256,256,256)):
        warnings.warn("Given shape {} is not as expected.".format(image_data.shape))
#        raise AssertionError ("shape is not as expected.")
    print("DATA_TYPE:",image_data.dtype)
    if (image_data.dtype is not 'uint8'):
        warnings.warn('Data type ({}) is not "uint8"'.format(image_data.dtype))
    return image_data
            
    
def scanFolder (folder_loc):
    "Returns a list of files in the folder."
    stack=[]
    print(type(os.walk(folder_loc)), os.walk(folder_loc))
    for path,sub_folders,files in os.walk(folder_loc):
        for file in files:
            stack.append(str(file))
        break;
    if len(stack)==0:
        raise AssertionError("No files in the folder.")
    return stack


def normalizeImage(nd_array,scale_down='range'):
    """returns a tuple: (mean,std,normalized_array), scaled down using the range."""
    mean=np.mean(nd_array)
    if scale_down=='range':
        nd_array=(nd_array-mean)/(np.max(nd_array)-np.min(nd_array))
    elif scale_down=='std':
        nd_array=(nd_array-mean)/np.std(nd_array)
    else:
        raise AssertionError ('Invalid scale_down var viz.',scale_down)
    
    return (nd_array)

def formatRange (nd_array,frm=0,to=1):
    """Brings all elements of an nd_array in the range [frm,to]"""
    nd_array=normalizeImage(nd_array)
    nd_array=frm+(to-frm)*((nd_array-np.min(nd_array))/(np.max(nd_array)-np.min(nd_array)))
    return nd_array



def viewImagesInIMGFile (img_loc):
    """View images in a hdr file. #CSF"""
    image_data=readFileAsNDArray(img_loc)
    image_data=(formatRange(image_data,0,255)).astype('uint8')
    print("Input axis== -1 to exit.")
    axis='2'
    while True:
        axis=input("Enter axis: ")
        if axis=='0' or axis=='1' or axis=='2':
            slice_num=int(input("Enter Slice Number: "))
            if axis=='0':
                image=image_data[slice_num,:,:]
            elif axis=='1':
                image=image_data[:,slice_num,:]
            else:
                image=image_data[:,:,slice_num]
            cv.imshow("Image_"+str(axis)+"_"+str(slice_num),image)
            cv.waitKey(0)
        elif axis=='-1':
            break
        else:
            raise ValueError("Invalid value given for axis.")
            
def checkConsistencyOfData(ndarray):
    for x in np.nditer(ndarray,order='F'):
        if (x!=0 and x!=10 and x!=150 and x!=250):
            raise ValueError

def generateMask(_2d_array,compare_value,low,high):
    _2d_array=np.copy(_2d_array)
    bool_mask=(_2d_array==compare_value)
    _2d_array[bool_mask]=high
    _2d_array[np.logical_not(bool_mask)]=low
    return _2d_array

        

def splitArray (_2d_array,compare_values=[0,10,150,250],low=0,high=1):
    split_list=[]
    if len(compare_values)!=4:
        raise AssertionError("Number of masks won't be 4.")
    for i in compare_values:
        split_list.append(generateMask(_2d_array,i,low,high))
    return split_list

def get4ChannelArray(image):
	"""Pass an array of shape (l,b,1) as input"""
	assert(len(image.shape)==3)
	assert(image.shape[2]==1)
	return np.concatenate(splitArray(image),axis=2)

def saveSlicedArrays (_3d_array,hdf_file_object,dataset_name,axis,from_idx,to_idx):
    """ Saves array sliced along the given "axis" to a datset(datset_name) in a HDF file.
        INPUT:_3d_array: Volume containing scan data. 
            hdf_file_object, dataset_name: A valid dataset name that is already present in the hdf file.
        axis: Takes values 0(Saggital),1(Coronal),or 2(Axial).
        from_idx: save the first slice at from_idx in the datset.
        to_idx: last slice will be saved at his idx in the dataset."""
    num_slices= to_idx-from_idx+1
    assert(num_slices==_3d_array.shape[axis])
    #TODO_ test this function on custom arrays 
    if axis==0:
        for i in range(num_slices):
            hdf_file_object[dataset_name][from_idx+i]=(cv.resize(_3d_array[i,:,:].reshape(256,256,1),
                           (LENGTH,WIDTH),interpolation=cv.INTER_NEAREST)).reshape(LENGTH,WIDTH,1)
    elif axis==1:
        for i in range(num_slices):
            hdf_file_object[dataset_name][from_idx+i]=(cv.resize(_3d_array[:,i,:].reshape(256,256,1),
                           (LENGTH,WIDTH),interpolation=cv.INTER_NEAREST)).reshape(LENGTH,WIDTH,1)
    elif axis==2:
        for i in range(num_slices):
            hdf_file_object[dataset_name][from_idx+i]=(cv.resize(_3d_array[:,:,i].reshape(256,256,1),
                           (LENGTH,WIDTH),interpolation=cv.INTER_NEAREST)).reshape(LENGTH,WIDTH,1)

def alignSaggital(_2d_array):
    _2d_array=np.rot90(_2d_array,1)
    _2d_array=np.flip(_2d_array,1)
    return _2d_array


def saveSlicedLabelArrays (_3d_array,hdf_file_object,dataset_name,axis,from_idx,to_idx):
    """ Saves array sliced along the given "axis" to a datset(datset_name) in a HDF file.
        INPUT:_3d_array: Volume containing scan data. 
            hdf_file_object, dataset_name: A valid dataset name that is already present in the hdf file.
        axis: Takes values 0(Saggital),1(Coronal),or 2(Axial).
        from_idx: save the first slice at from_idx in the datset.
        to_idx: last slice will be saved at his idx in the dataset."""
    num_slices= to_idx-from_idx+1
    assert(num_slices==_3d_array.shape[axis])
    #DONE_ test this function on custom arrays 
    if axis==0:
        for i in range(num_slices):
            hdf_file_object[dataset_name][from_idx+i]=get4ChannelArray((cv.resize((alignSaggital(_3d_array[i,:,:])).reshape(256,256,1),
                           (LENGTH,WIDTH),interpolation=cv.INTER_NEAREST)).reshape(LENGTH,WIDTH,1))
    elif axis==1:
        #NOTE_: 1 and 2 are interchanged
        for i in range(num_slices):
            hdf_file_object[dataset_name][from_idx+i]=get4ChannelArray((cv.resize((np.flip(_3d_array[:,:,i],1)).reshape(256,256,1),
                           (LENGTH,WIDTH),interpolation=cv.INTER_NEAREST)).reshape(LENGTH,WIDTH,1))
    elif axis==2:
        for i in range(num_slices):
            hdf_file_object[dataset_name][from_idx+i]=get4ChannelArray((cv.resize((np.flip(_3d_array[:,i,:])).reshape(256,256,1),
                           (LENGTH,WIDTH),interpolation=cv.INTER_NEAREST)).reshape(LENGTH,WIDTH,1))

    
#TODO_ add storage for all axis.. possible change in num samples. Add assertions fro size check..    

def createTrainingDataset(root_folder,op_folder,filename):
    stk=scanFolder(root_folder)
    total_samples=0
    for items in stk:
        if ".mgz" in items:
            total_samples+=1
    assert(total_samples%3==0)
    num=D_LENGTH*total_samples/3
    
    
    with hf.File(op_folder+"/"+filename+".h5",'w') as f:
        T1=f.create_dataset("T1",shape=(num,LENGTH,WIDTH,1))
        T2=f.create_dataset("T2",shape=(num,LENGTH,WIDTH,1))
        label=f.create_dataset("label",shape=(num,LENGTH,WIDTH,4))
        ignore_count=0
        saved_count=0
        for item in stk:
            
            if ".mgz" in item:
                saved_count+=1
                print("-"*10,saved_count,"-"*10)
                s=item.split("-")#s[1] is subject number
                subject_num=int(s[1])
                frm=(subject_num-1)*D_LENGTH
                to=frm+255
                print("from:",frm,"to:",to)
                
                if s[2]=='T1.img.mgz':
                    saveSlicedArrays(readFileAsNDArray(root_folder+"/"+item),
                                     f,"T1",1,frm,to)
                elif s[2]=='T2.img.mgz':
                    saveSlicedArrays(readFileAsNDArray(root_folder+"/"+item),
                                     f,"T2",1,frm,to)
                elif s[2]=='label.img.mgz':
                    saveSlicedLabelArrays(readFileAsNDArray(root_folder+"/"+item),
                                     f,"label",1,frm,to)
                else:
                    raise ValueError("subject_num is invalid")
                print(item)
                print("*"*20)
            else:
                ignore_count+=1
                print(ignore_count,"Ignored:" , item)
            
def createTestDataset(root_folder,op_folder,filename,lowest_subject_number=11):
    stk=scanFolder(root_folder)
    total_samples=0
    for items in stk:
        if ".img.mgz" in items:
            total_samples+=1
    assert(total_samples%2==0)
    num=D_LENGTH*total_samples/2
    
    
    with hf.File(op_folder+"/"+filename+".h5",'w') as f:
        T1=f.create_dataset("test_T1",shape=(num,LENGTH,WIDTH,1))
        T2=f.create_dataset("test_T2",shape=(num,LENGTH,WIDTH,1))
        ignore_count=0
        saved_count=0
        for item in stk:
            
            if ".img.mgz" in item:
                saved_count+=1
                print("-"*10,saved_count,"-"*10)
                print(item)
                s=item.split("-")#s[1] is subject number
                subject_num=int(s[1])
                frm=(subject_num-1-lowest_subject_number)*D_LENGTH
                to=frm+255
                if s[2]=='T1.img.mgz':
                    saveSlicedArrays(readFileAsNDArray(root_folder+"/"+item),
                                     f,"test_T1",1,frm,to)
                elif s[2]=='T2.img.mgz':
                    saveSlicedArrays(readFileAsNDArray(root_folder+"/"+item),
                                     f,"test_T2",1,frm,to)
                else:
                    raise ValueError("subject_num is invalid")
                
                print("*"*20)
            else:
                ignore_count+=1
                print(ignore_count,"Ignored:" , item)

def getNextBatch(file_object,dataset_type='training',batch_size=1,dtype='float32',verbose=True,shuffle=True):
    if dataset_type=='training' or dataset_type=='train':
        if shuffle:
            X,Y=get_next_training_batch(file_object,batch_size,verbose)
        else:
            X,Y=getNextTrainingBatch(file_object,batch_size,verbose)
        return (X.astype(dtype),Y.astype(dtype))
    elif dataset_type=='testing' or dataset_type=='test':
        X,Y=getNextTestBatch(file_object,batch_size,verbose)
        return (X.astype(dtype),Y.astype(dtype))
    else:
        raise ValueError('The value given for dataset_type (',dataset_type,
        ') is invalid.')

def merge4ChannelArray(nd_array):
    assert (nd_array.shape==(LENGTH,WIDTH,4))
    mask=np.argmax(nd_array,axis=2)
    y=np.zeros(shape=(LENGTH,WIDTH),dtype=np.uint8)
    for i in range(len(CHANNEL_VALUES)): 
        y[mask==i]=CHANNEL_VALUES[i]
    return y

def getNextTrainingBatch (f,batch_size=10,verbose=True):
    if verbose:
        print("-"*5,"getting next batch for training.","-"*5)
    if ('count' not in getNextTrainingBatch.__dict__) or (getNextTrainingBatch.f != f):
        getNextTrainingBatch.count=1
        getNextTrainingBatch.f=f
        getNextTrainingBatch.T1=getNextTrainingBatch.f['T1']
        getNextTrainingBatch.T2=getNextTrainingBatch.f['T2']
        getNextTrainingBatch.label=getNextTrainingBatch.f['label']
        getNextTrainingBatch.shape=getNextTrainingBatch.T1.shape
        getNextTrainingBatch.begin=0
        getNextTrainingBatch.end=batch_size
        if verbose:
            print("*"*5,"Values_Reinitialized","*"*5)
    
    #Checking consistency###################
    if (batch_size>getNextTrainingBatch.shape[0]):
        raise AssertionError("Batch size exceeds the size of datset.")   
    ##########Compute Indices###############
    if getNextTrainingBatch.count>1:
        getNextTrainingBatch.begin=(getNextTrainingBatch.end+1)%(getNextTrainingBatch.shape[0])
        
    getNextTrainingBatch.end=(getNextTrainingBatch.begin+batch_size-1)%(getNextTrainingBatch.shape[0])
    is_split=(getNextTrainingBatch.end<getNextTrainingBatch.begin)
    
    frm=getNextTrainingBatch.begin
    max_idx=(getNextTrainingBatch.shape[0]-1)
    end=getNextTrainingBatch.end
    
    if is_split:
        
        T1=np.concatenate([getNextTrainingBatch.T1[frm:max_idx+1],
                                        getNextTrainingBatch.T1[0:end+1]],axis=0)
        T2=np.concatenate([getNextTrainingBatch.T2[frm:max_idx+1],
                                        getNextTrainingBatch.T2[0:end+1]],axis=0)
        label=np.concatenate([getNextTrainingBatch.label[frm:max_idx+1],
                                        getNextTrainingBatch.label[0:end+1]],axis=0)
    else:
        T1=getNextTrainingBatch.T1[frm:end+1]
        T2=getNextTrainingBatch.T2[frm:end+1]
        label=getNextTrainingBatch.label[frm:end+1]
    
    X_train=np.concatenate([T1,T2],axis=3)
    Y_train=label
    
    if verbose:
        print ("#"*5,"BATCH_NUM=",getNextTrainingBatch.count,'#'*5)
        if is_split:
            print("[",str(frm),",",str(max_idx)+"] U [",0,",",end,"]")
        else:
            print("[",str(frm),", ",str(end),"]")
        print("f=",getNextTrainingBatch.f)
        print('begin=',getNextTrainingBatch.begin)
        print('end=',getNextTrainingBatch.end)
        print('X_train:',X_train.shape,X_train.dtype)
        print('Y_train:',Y_train.shape,Y_train.dtype)
    
    assert(X_train.shape==(batch_size,LENGTH,WIDTH,2))
    assert(Y_train.shape==(batch_size,LENGTH,WIDTH,NUM_OP_CHANNELS))
    

    getNextTrainingBatch.count+=1
    return (X_train,Y_train)
        
def getNextTestBatch (f,batch_size=10,verbose=True):
    if verbsoe:
        print("-"*5,"getting next batch for testing.","-"*5)
    if ('count' not in getNextTestBatch.__dict__) or (getNextTestBatch.f != f):
        getNextTestBatch.count=1
        getNextTestBatch.f=f
        getNextTestBatch.T1=getNextTestBatch.f['T1']
        getNextTestBatch.T2=getNextTestBatch.f['T2']
        getNextTestBatch.shape=getNextTestBatch.T1.shape
        getNextTestBatch.begin=0
        getNextTestBatch.end=batch_size
        if verbose:
            print("*"*5,"Values_Reinitialized","*"*5)
    
    #Checking consistency###################
    if (batch_size>getNextTestBatch.shape[0]):
        raise AssertionError("Batch size exceeds the size of datset.")   
    ##########Compute Indices###############
    if getNextTestBatch.count>1:
        getNextTestBatch.begin=(getNextTestBatch.end+1)%(getNextTestBatch.shape[0])
        
    getNextTestBatch.end=(getNextTestBatch.begin+batch_size-1)%(getNextTestBatch.shape[0])
    is_split=(getNextTestBatch.end<getNextTestBatch.begin)
    
    frm=getNextTestBatch.begin
    max_idx=(getNextTestBatch.shape[0]-1)
    end=getNextTestBatch.end
    
    if is_split:
        
        T1=np.concatenate([getNextTestBatch.T1[frm:max_idx+1],
                                        getNextTestBatch.T1[0:end+1]],axis=0)
        T2=np.concatenate([getNextTestBatch.T2[frm:max_idx+1],
                                        getNextTestBatch.T2[0:end+1]],axis=0)
    else:
        T1=getNextTestBatch.T1[frm:end+1]
        T2=getNextTestBatch.T2[frm:end+1]
    
    X_test=np.concatenate([T1,T2],axis=3)
    
    assert(X_test.shape==(batch_size,LENGTH,WIDTH,2))
    if verbose:
        print ("#"*5,"BATCH_NUM=",getNextTestBatch.count,'#'*5)
        if is_split:
            print("[",str(frm),",",str(max_idx)+"] U [",0,",",end,"]")
        else:
            print("[",str(frm),", ",str(end),"]")
        print("f=",getNextTestBatch.f)
        print('begin=',getNextTestBatch.begin)
        print('end=',getNextTestBatch.end)
        print('X_train:',X_test.shape,X_test.dtype)
    getNextTestBatch.count+=1
    return (X_test,)
    
def get_chunk (file_object,dataset_name,from_,to_,verbose=True,dtype='float32'):
    X=file_object[dataset_name]
    max_idx=X.shape[0]-1
    
    if(from_==to_):
        raise AssertionError("from_ and to_ should be unequal.")
    elif(from_>to_):
        batch_size=X.shape[0]+from_-to_-1
        assert(batch_size<=X.shape[0])
        X_chunk=(np.concatenate([X[from_:max_idx+1],X[0:to_+1]],
                               aixs=0)).astype(dtype)
    else:
        batch_size=to_-from_+1
        assert(batch_size<=X.shape[0])
        X_chunk=X[from_:to_+1]
    if verbose:
        print("Datset: ",dataset_name,"dtype:",X.dtype)
        print("From:",from_,"to:",to_)
        print("chunk.shape",X_chunk.shape)
        print("chunk.dtype:",X_chunk.dtype)
    
    return X_chunk

def getGenerator(hdf_file_loc,mode='train',shuffle=True,batch_size=10,dtype=np.float32):
    f=hf.File(hdf_file_loc,'r')
    def generator():
        generator.file=f
        T1=f['T1']
        T2=f['T2']
        label=f['label']
        size=T1.shape[0]
        assert(batch_size<=T1.shape[0])
        print("size",size,batch_size)
        #assert(size%batch_size==0)
        count=0
        while True:
            count+=1
            print('-'*8,str(count),"reached here")
            dummy=input("Hye")
            X,Y=getNextBatch(f,'train',batch_size,shuffle=shuffle)
            dummy=input("Bye") 
            if IMAGE_DIM_ORDERING=='tf':
                yield X,Y
            elif IMAGE_DIM_ORDERING=='th':
                yield X.transpose(0,3,1,2),Y.transpose(0,3,1,2)
            else:
                raise ValueError()
            print('-'*8,str(count),"reached here too")
            dummy=input("BYye")
    return generator



def get_sample(file_object,dataset_name,list_of_indices,verbose=True):
    X=file_object[dataset_name]
    list_of_samples=[]
    if dataset_name=='T1' or dataset_name=='T2':
        channels=1
    elif dataset_name=='label':
        channels=4
    else:
        raise ValueError("dataset_name should be one of: T1 T2 or label")
    
    for index in list_of_indices:
        list_of_samples.append((X[index]).reshape(1,LENGTH,WIDTH,channels))
    
    X_chunk=np.concatenate(list_of_samples,axis=0)
    
    if verbose:
        print("-----Dataset:",dataset_name,"-"*5)
        print("Indices:",list_of_indices)
        print(dataset_name,":",X.shape,X.dtype)
        print("Chunk:",X_chunk.shape,X_chunk.dtype)
    if IMAGE_DIM_ORDERING=='tf':
        return X_chunk
    elif IMAGE_DIM_ORDERING=='th':
        return X_chunk.transpose(0,3,1,2)
    else:
        raise ValueError()


def get_next_training_batch (f,batch_size=10,verbose=True):
    if verbose:
        print("-"*5,"getting next batch for training.","-"*5)
    if ('count' not in get_next_training_batch.__dict__) or (get_next_training_batch.f != f):
        get_next_training_batch.count=1
        get_next_training_batch.f=f
        get_next_training_batch.T1=get_next_training_batch.f['T1']
        get_next_training_batch.T2=get_next_training_batch.f['T2']
        get_next_training_batch.label=get_next_training_batch.f['label']
        get_next_training_batch.shape=get_next_training_batch.T1.shape
        get_next_training_batch.list_of_indices=np.arange(get_next_training_batch.shape[0])
        np.random.shuffle(get_next_training_batch.list_of_indices)
        get_next_training_batch.begin=0
        get_next_training_batch.end=batch_size
        if verbose:
            print("*"*5,"Values_Reinitialized","*"*5)
    
    #Checking consistency###################
    if (batch_size>get_next_training_batch.shape[0]):
        raise AssertionError("Batch size exceeds the size of datset.")   
    ##########Compute Indices###############
    if get_next_training_batch.count>1:
        get_next_training_batch.begin=(get_next_training_batch.end+1)%(get_next_training_batch.shape[0])
        
    get_next_training_batch.end=(get_next_training_batch.begin+batch_size-1)%(get_next_training_batch.shape[0])
    is_split=(get_next_training_batch.end<get_next_training_batch.begin)
    
    frm=get_next_training_batch.begin
    max_idx=(get_next_training_batch.shape[0]-1)
    end=get_next_training_batch.end
    
    if is_split:
        
        list_of_indices=np.concatenate([get_next_training_batch.list_of_indices[frm:max_idx+1],
                                        get_next_training_batch.list_of_indices[0:end+1]],axis=0)
    else:
        list_of_indices=get_next_training_batch.list_of_indices[frm:end+1]
    
    T1=get_sample(f,'T1',list_of_indices)
    T2=get_sample(f,'T2',list_of_indices)    
    label=get_sample(f,'label',list_of_indices)
    X_train=np.concatenate([T1,T2],axis=3)
    Y_train=label
    
    if verbose:
        print ("#"*5,"BATCH_NUM=",get_next_training_batch.count,'#'*5)
        if is_split:
            print("[",str(frm),",",str(max_idx)+"] U [",0,",",end,"]")
        else:
            print("[",str(frm),", ",str(end),"]")
        print("f=",get_next_training_batch.f)
        print('begin=',get_next_training_batch.begin)
        print('end=',get_next_training_batch.end)
        print('X_train:',X_train.shape,X_train.dtype)
        print('Y_train:',Y_train.shape,Y_train.dtype)
    
    assert(X_train.shape==(batch_size,LENGTH,WIDTH,2))
    assert(Y_train.shape==(batch_size,LENGTH,WIDTH,NUM_OP_CHANNELS))
    

    get_next_training_batch.count+=1
    return (X_train,Y_train)


def normalize_array(array):
    return array/255

def denormalize_array(array):
    return array*255

def create_smaller_dataset(data_loc,chunks=None,suffix='_'):
    with hf.File(data_loc,'r') as f:
        name=data_loc[:-3]
        img_shape=f['T1'].shape
        num_samples=0
        for item in chunks:
            num_samples = num_samples + (item[1]-item[0]+1)
        input_shape=(num_samples,img_shape[1],img_shape[2],1)
        label_shape=(num_samples,img_shape[1],img_shape[2],4)
        
        a=hf.File(name+suffix+'.h5','w')
        a.create_dataset('T1',shape=input_shape,dtype='float32')
        a.create_dataset('T2',shape=input_shape,dtype='float32')
        a.create_dataset('label',shape=label_shape,dtype='float32')
        
        print('Given_dataset : ',img_shape)
        print('Partition : ', input_shape)
        idx=0
        chunk_num=1
        for item in chunks:
            chunk_size=item[1]-item[0]+1
            print('Chunk:',chunk_num,item,'->','[',idx,',',idx+chunk_size-1,']')
            for i in range(chunk_size):
                a['T1'][idx]=f['T1'][item[0]+i]
                a['T2'][idx]=f['T2'][item[0]+i]
                a['label'][idx]=f['label'][item[0]+i]
                idx= idx+1
            chunk_num+=1
        a.close()

def partition (dataset_loc,chunks_train,chunks_val,suffix=['_','_']):
    print("Training_dataset\n")
    create_smaller_dataset(dataset_loc,chunks_train,'_train'+suffix[0])
    print('validation_dataset.\n')
    create_smaller_dataset(dataset_loc,chunks_val,'_val'+suffix[1])


#%%Data Generator Class
class Generator:
    def __init__(self,dataset_locs=[],batch_size=10):
        self.batch_size=batch_size
        self.files=[]
        for items in dataset_locs:
            self.files.append(hf.File(items,'r'))
        
        self.sizes=[]
        for file in self.files:
            sizes.append(file['T1'].shape[0])
        
        self.num_samples=sum(self.sizes)
        self.max_idx=self.num_samples-1
        self.X1,self.X2,self.Y=self.load_data()
        #data associated with class
        self.X1_batch = None
        self.X2_batch = None
        self.Y_batch = None
        
        #Counters to maintain batch synchronism
        self.current_batch=0
        self.num_of_batches= int(self.num_samples/self.batch_size)+1
        self.extra_required= (self.batch_size
                                    - self.num_samples%self.batch_size)
        
    def load_data(self):
        i=0
        X1=None
        X2=None
        Y=None
        for f in self.files:
            if i==0:
                X1=f['T1'][:]
                X2=f['T2'][:]
                Y=f['label'][:]
            else:
                X1=np.concatenate([X1,f['T1'][:]],axis=0)
                X2=np.concatenate([X2,f['T2'][:]],axis=0)
                Y=np.concatenate([Y,f['label'][:]],axis=0)
            i+=1
            f.close()
        return (X1,X2,Y)
    
    def shuffle_all(self):
        rnd_state = numpy.random.get_state()
        np.random.shuffle(self.X1)
        numpy.random.set_state(rnd_state)
        np.random.shuffle(self.X2)
        numpy.random.set_state(rnd_state)
        np.random.shuffle(self.Y)
    
    def get_next_batch(self):
        #Note: current_batch ranges from 0 to num_batches-1
        if (self.current_batch==(self.num_of_batches-1)):
            start_idx=self.current_batch*self.batch_size
            extra=np.random.randint(0,self.num_samples,self.extra_required)
            self.X1_batch=np.copy(np.concatenate([self.X1[start_idx:self.max_idx+1],
                                             self.X1[extra]]))
            self.X2_batch=np.copy(np.concatenate([self.X2[start_idx:self.max_idx+1],
                                             self.X2[extra]]))
            self.Y_batch = np.copy(np.concatenate([self.Y[start_idx:self.max_idx+1],
                                              self.Y[extra]]))
            self.shuffle_all()
        else:
            start_idx=self.current_batch*self.batch_size
            end_idx=start_idx+self.batch_size-1
            self.X1_batch=np.copy(self.X1[start_idx:end_idx+1])
            self.X2_batch=np.copy(self.X2[start_idx:end_idx+1])
            self.Y_batch = np.copy(self.Y[start_idx:end_idx+1])
            
        self.current_batch= (self.current_batch+1)%(self.num_of_batches)
        
        
            
    
    def augment_current_batch(self):
        pass
    
    def get_generator(self):
        return self.data_generator
    
    def data_generator(self):
        self.current_batch+=1
#        while True:
#            self.get_next_batch()
#            self.augment_current_batch()
#            
#            yield (np.concatenate([self.X1_batch,self.X2_batch], axis=0),
#                                  self.Y_batch)
    def summary(self):
        print(self.__dict__)



#%%SCANNING
def scanTrainingDataset(root_folder):
    stk=scanFolder(root_folder)
    total_samples=0
    for items in stk:
        if ".mgz" in items:
            total_samples+=1
    assert(total_samples%3==0)
    num = D_LENGTH*total_samples/3

    #T1=f.create_dataset("T1",shape=(num,LENGTH,WIDTH,1))
#        T2=f.create_dataset("T2",shape=(num,LENGTH,WIDTH,1))
#        label=f.create_dataset("label",shape=(num,LENGTH,WIDTH,4))
    ignore_count=0
    saved_count=0
    for item in stk:
        
        if ".img" in item:
            saved_count+=1
            print("-"*10,saved_count,"-"*10)
            s=item.split("-")#s[1] is subject number
            subject_num=int(s[1])
            location = root_folder+"/"+item
            if s[2]=='T1.img':
                print('Location:',location,s[2])
                shape = readFileAsNDArray(root_folder+"/"+item).shape
                print(shape)
                if not (shape == (144,192,256,1)):
                    x=input('Shape Not as expected')
#                assert(shape==(144,192,256,1))
            elif s[2]=='T2.img':
                print('Location:',location,s[2])
                shape = readFileAsNDArray(root_folder+"/"+item).shape
                print(shape)
                if not (shape == (144,192,256,1)):
                    x=input('Shape Not as expected')
#                assert(shape==(144,192,256,1))
                 
            elif s[2]=='label.img':
                print('Location:',location,s[2])
                shape = readFileAsNDArray(root_folder+"/"+item).shape
                print(shape)
                if not (shape == (144,192,256,1)):
                    x=input('Shape Not as expected')
#                assert(shape==(144,192,256,1))
                 
            else:
                raise ValueError("subject_num is invalid")
            print(item)
            print("*"*20)
        else:
            ignore_count+=1
            print(ignore_count,"Ignored:" , item)
            
#%%Create PAdded 3D Volumes:
def padVolume (input_vol):
    shape = input_vol.shape
    dtype = input_vol.dtype
    drow_up = (256-input_vol.shape[0])//2
    drow_dn = 256-input_vol.shape[0]-drow_up
    dcol_lt = (256-input_vol.shape[1])//2
    dcol_rt = 256-input_vol.shape[1]-dcol_lt
    ddepth_front = (256-input_vol.shape[2])//2
    ddepth_back = 256-input_vol.shape[2]-ddepth_front
    
    print ('Given Shape:',shape)
    print ('Padding :',drow_up,drow_dn,dcol_lt,dcol_rt,ddepth_front,ddepth_back)
    #UP
    pad = np.zeros((drow_up,shape[1],shape[2],shape[3]), dtype)
    input_vol = np.concatenate([pad,input_vol],axis=0)
    #DOWN
    pad = np.zeros((drow_dn,shape[1],shape[2],shape[3]), dtype)
    input_vol = np.concatenate([input_vol,pad],axis=0)
    #Left
    pad = np.zeros((256,dcol_lt,shape[2],shape[3]), dtype)
    input_vol = np.concatenate([pad,input_vol],axis=1)
    #DOWN
    pad = np.zeros((256,dcol_rt,shape[2],shape[3]), dtype)
    input_vol = np.concatenate([input_vol,pad],axis=1)
    #Front
    pad = np.zeros((256,256,ddepth_front,shape[3]), dtype)
    input_vol = np.concatenate([pad,input_vol],axis=2)
    #Back
    pad = np.zeros((256,256,ddepth_back,shape[3]), dtype)
    input_vol = np.concatenate([input_vol, pad],axis=2)
    print('Output Shape:',input_vol.shape)
    return input_vol
    
    

def createPaddedVolumes (root_folder, DATA_TYPE):
    stk=scanFolder(root_folder)
    total_samples=0
    for items in stk:
        if items[-4:]=='.img':
            total_samples+=1
    print('Total Volumes in the folder:', total_samples)
    
    ignore_count=0
    saved_count=0
    for item in stk:
        
        if item[-4:] == '.img':
            saved_count+=1
            print("-"*10,saved_count,"-"*10)
            s=item.split("-")#s[1] is subject number
            subject_num=int(s[1])
            location = root_folder+"/"+item
            
            vol_type = (s[2].split('.'))[0]
            print('Volume_type:',vol_type)
            print('Location:',location,s[2])
            x = readFileAsNDArray(location)
            if x.dtype != DATA_TYPE:
                print('----changing data type to',DATA_TYPE)
                x = x.astype(DATA_TYPE)
                print('DTYPE OVERIDDEN')
            shape =x.shape
            print('Shape:',shape)
            if not (shape == (144,192,256,1)):
                x=input('Shape Not as expected')
            
            file_name = 'sub-'+str(subject_num)+'-'+vol_type+'.h5'
            with hf.File(TRAIN_FOLDER+'/'+file_name,'w') as f:
                f.create_dataset(vol_type,data = padVolume(x))
        else:
            ignore_count+=1
            print(ignore_count,"Ignored:" , item)
def getFormattedT1orT2 (inp,axis):
    if axis==0:
        return inp
    elif axis==1:
        return inp.transpose(1,0,2,3)
    elif axis==2:
        return inp.transpose(2,1,0,3)
    else:
        raise ValueError('Invalid axis:',axis)

def getFormattedLabel (inp,axis):
    shape = list(inp.shape)
    assert(shape[3]==1)
    assert(len(shape)==4)
    shape[3] = 4
    x = np.zeros(tuple(shape),dtype=inp.dtype)
    for i in range(256):
        x[:,:,i,:] = get4ChannelArray(inp[:,:,i:i+1,0])
    
    if axis==0:
        return x
    elif axis==1:
        return x.transpose(1,0,2,3)
    elif axis==2:
        return x.transpose(2,1,0,3)
    else:
        raise ValueError('Invalid axis:',axis)
    

            
def getFormattedDataset (inp, vol_type, axis):
    if vol_type == 'T1' or vol_type=='T2':
        return getFormattedT1orT2(inp,axis)
    elif vol_type == 'label':
        return getFormattedLabel(inp,axis)

def createPaddedDatasets (root_folder, DATA_TYPE,axis):
    stk=scanFolder(root_folder)
    total_samples=0
    for items in stk:
        if items[-4:]=='.img':
            total_samples+=1
    print('Total Volumes in the folder:', total_samples)
    
    ignore_count=0
    saved_count=0
    for item in stk:
        
        if item[-4:] == '.img':
            saved_count+=1
            print("-"*10,saved_count,"-"*10)
            s=item.split("-")#s[1] is subject number
            subject_num=int(s[1])
            location = root_folder+"/"+item
            
            vol_type = (s[2].split('.'))[0]
            print('Volume_type:',vol_type)
            print('Location:',location,s[2])
            x = readFileAsNDArray(location)
            if x.dtype != DATA_TYPE:
                print('----changing data type to',DATA_TYPE)
                x = x.astype(DATA_TYPE)
                print('DTYPE OVERIDDEN')
            shape =x.shape
            print('Shape:',shape)
            if not (shape == (144,192,256,1)):
                x=input('Shape Not as expected')
            
            file_name = 'subject-'+str(subject_num)+'-axis-'+str(axis)+'.h5'
            with hf.File(TRAIN_FOLDER+'/'+file_name,'a') as f:
                print('Creating',vol_type,'in',file_name)
                f.create_dataset(vol_type,data = getFormattedDataset(padVolume(x),vol_type,axis))
        else:
            ignore_count+=1
            print(ignore_count,"Ignored:" , item)
            
#%%FOR GETTING BACK ORIGINAL VERSION OF DATA
def get_back_original_T1orT2 (inp,axis):
    return getFormattedT1orT2(inp,axis)
    
def get_back_label_with_original_orientation(inp,axis):
    'Give four channel label map tensor as input.'
    if axis==0:
        return inp
    elif axis==1:
        return inp.transpose(1,0,2,3)
    elif axis==2:
        return inp.transpose(2,1,0,3)
    else:
        raise ValueError('Invalid axis:',axis)
    
    
    
def get_back_original_label(inp,axis):
    """INput: A channel tensor : (256,256,256,4)OUtput: A (256,256,256,1)"""
    inp = get_back_label_with_original_orientation(inp,axis)
    shape = list(inp.shape)
    shape[3]=1
    assert (shape == (256,256,256,1))
    op = np.zeros(shape,dtype=inp.dtype)
    for i in range(shape[2]):
        op[:,:,i,0]=merge4ChannelArray(inp[:,:,i,:])
    return op
        
def generate_original_volumes(file_list):
    """Give absolute paths in the file_list"""
    print('Not yet tested\n\n\n')
    for i in range(len(file_list)):
        with hf.File(file_list[i],'r') as f:
            path_parts = file_list[i].split(os.sep)
            name_parts = path_parts[-1].split('-')
            current_file = file_list[i]
            out_file = current_file[:-3]+'_inverted.h5'
            axis =int(name_parts[3].split('.')[0])
            g= hf.File(out_file,'w')
            if 'T1' in f.keys():
                t1= f['T1'][:]
                t1_ = get_back_original_T1orT2(t1,axis)
                g.create_dataset('T1',data=t1_)
                del t1_
                del t1
            
            if 'T2' in f.keys():
                t2= f['T2'][:]
                t2_ = get_back_original_T1orT2(t2,axis)
                g.create_dataset('T2',data=t2_)
                del t2_
                del t2
                
            if 'label' in f.keys():
                lab= f['label'][:]
                lab_ = get_back_label_with_original_orientation(lab,axis)
                g.create_dataset('label',data=lab_)
                del lab_
                del lab
            
            g.close()


if __name__ == '__main__':
    rfolder = '/home/shubham/Desktop/IMG'
#    scanTrainingDataset('/media/shubham/37b9890f-9a79-41f6-8ccd-50e1d7a93461/BCP_BABY_BRAIN_test')
#    y = readFileAsNDArray('/media/shubham/37b9890f-9a79-41f6-8ccd-50e1d7a93461/BCP_BABY_BRAIN_test/subject-23-T2.img')
##    y= readFileAsNDArray('/home/shubham/Desktop/BCP/Datasets/RAW_Test/subject-23-T2.img.mgz')
#    y =padVolume(y)
#    createPaddedDatasets (rfolder,'float32',0)
#    createPaddedDatasets (rfolder,'float32',1)
#    createPaddedDatasets (rfolder,'float32',2)
  #%% for generating originals  
#    file_list = ['subject-1-axis-0.h5', 'subject-1-axis-1.h5', 'subject-1-axis-2.h5']
#    
#    for i in range(len(file_list)):
#        with hf.File(TRAIN_FOLDER+'/'+file_list[i],'r') as f:
#            name_parts = file_list[i].split('-')
#            current_file = TRAIN_FOLDER+'/'+file_list[i]
#            out_file = current_file[:-3]+'_inverted.h5'
#            axis =int(name_parts[3].split('.')[0])
#            g= hf.File(out_file,'w')
#            t1= f['T1'][:]
#            t1_ = get_back_original_T1orT2(t1,axis)
#            g.create_dataset('T1',data=t1_)
#            del t1_
#            del t1
#            
#            t2= f['T2'][:]
#            t2_ = get_back_original_T1orT2(t2,axis)
#            g.create_dataset('T2',data=t2_)
#            del t2_
#            del t2
#            
#            lab= f['label'][:]
#            lab_ = get_back_label_with_original_orientation(lab,axis)
#            g.create_dataset('label',data=lab_)
#            del lab_
#            del lab
#            g.close()

#            g= hf.File(out_file,'w')
#            t1= f['T1'][:]
#            t1_ = get_back_original_T1orT2(t1,axis)
#            g.create_dataset('T1',data=t1_)
#            del t1_
#            del t1
#            
#            t2= f['T2'][:]
#            t2_ = get_back_original_T1orT2(t2,axis)
#            g.create_dataset('T2',data=t2_)
#            del t2_
#            del t2
#            
#            lab= f['label'][:]
#            lab_ = get_back_label_with_original_orientation(lab,axis)
#            g.create_dataset('label',data=lab_)
#            del lab_
#            del lab
#            g.close()

#    x = x.astype('float32')
#    print (x.shape)
#    np.set_printoptions(threshold = np.inf)
#    print (np.unique(x,return_counts = True))
#    
#    print('Y=====0')
#    print (y.shape)
#    np.set_printoptions(threshold = np.inf)
#    print (np.unique(y,return_counts = True))
#    while True:
#        axis = int(input('axis'))
#        slice_num = int(input('slice_num'))
#            
#        if axis == 0:
#            im = [x[slice_num:slice_num+1,:,:,0]]
#            img = []
#            img.append(im[0].transpose(1,2,0))
#            img.append(im[0].transpose(2,1,0))
#            print(img[0].shape, img[1].shape)
#        elif axis == 1:
#            im = [x[:,slice_num:slice_num+1,:,0]]
#            img = []
#            img.append(im[0].transpose(0,2,1))
#            img.append(im[0].transpose(2,0,1))
#        elif axis ==2:
#            img = [x[:,:,slice_num:slice_num+1,0]]
#        for i in range(len(img)):
#            print('Current Shape:',img[i].astype('uint8').shape)
#            cv.imshow('X'+str(i),img[i].astype('uint8'))
#            cv.waitKey(0)

      

    