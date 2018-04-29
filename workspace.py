#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:12:00 2017
workspace
@author: shubham
"""
from default import *
import data as dt
import skvideo.io


def CVid(vw,image_stack):
    image_stack=image_stack.astype(np.uint8)
    skvideo.io.vwrite("ax3.mp4",image_stack,verbosity=1)

def gbdl(array):
    row_min=0
    row_len=0
    row_min_found=False

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i,j]>0:
                row_min_found=True
                break
        if row_min_found:
            row_min=i
            break

    if row_min_found:
        for i in range(row_min+1,array.shape[0],1):
            count=0
            for j in range(array.shape[1]):
                if array[i,j]!=0:
                    count+=1
            if count>0:
                row_len+=1
            else:
                break

        for i in range(row_min+row_len+1,array.shape[0],1):
            for j in range(array.shape[1]):
                if array[i,j]>0:
                    print("Zero sum found after row_max","at Index:",i,j)
                    #dummy=input('csdc')
                    #raise AssertionError()

    col_min=0
    col_len=0
    col_min_found=False

    for i in range(array.shape[1]):
        for j in range(array.shape[0]):
            if array[j,i]>0:
                col_min_found=True
                break
        if col_min_found:
            col_min=i
            break

    if col_min_found:
        for i in range(col_min+1,array.shape[1],1):
            count=0
            for j in range(array.shape[1]):
                if array[j,i]!=0:
                    count+=1
            if count>0:
                col_len+=1
            else:
                break
        for i in range(col_min+col_len+1,array.shape[1],1):
            for j in range(array.shape[1]):
                if array[j,i]>0:
                    print("Zero sum found after col_max","at Index:",i)
                    #dummy=input('x')
                    #raise AssertionError()

    return [row_min,col_min,row_len,col_len]#,row_min+row_len,col_min+col_len]

def gbdl2(array):
    row_min=0
    row_len=0
    row_min_found=False

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i,j]>0:
                row_min_found=True
                break
        if row_min_found:
            row_min=i
            break

    if row_min_found:
        for i in range(row_min+1,array.shape[0],1):
            count=0
            for j in range(array.shape[1]):
                if array[i,j]!=0:
                    count+=1
            if count>0:
                row_len+=1
            else:
                break

        for i in range(row_min+row_len+1,array.shape[0],1):
            for j in range(array.shape[1]):
                if array[i,j]>0:
                    print("Zero sum found after row_max","at Index:",i,j)
                    #dummy=input('csdc')
                    #raise AssertionError()

    col_min=0
    col_len=0
    col_min_found=False

    for i in range(array.shape[1]):
        for j in range(array.shape[0]):
            if array[j,i]>0:
                col_min_found=True
                break
        if col_min_found:
            col_min=i
            break

    if col_min_found:
        for i in range(col_min+1,array.shape[1],1):
            count=0
            for j in range(array.shape[1]):
                if array[j,i]!=0:
                    count+=1
            if count>0:
                col_len+=1
            else:
                break
        for i in range(col_min+col_len+1,array.shape[1],1):
            for j in range(array.shape[1]):
                if array[j,i]>0:
                    print("Zero sum found after col_max","at Index:",i)
                    #dummy=input('x')
                    #raise AssertionError()

    return [row_min,col_min,row_len,col_len]#,row_min+row_len,col_min+col_len]

def g(x):
    s=x.shape
    row_min=0
    row_max=s[0]-1
    col_min=0
    col_max=s[1]-1
    row_min_found=False
    row_max_found=False
    col_min_found=False
    col_max_found=False
    for i in range(s[0]):
        for j in range(s[1]):
            if x[i,j]>0:
                row_min_found=True
        if row_min_found:
            row_min=i
            break
    for i in range(s[0]):
        for j in range(s[1]):
            if x[row_max-i,j]>0:
                row_max_found=True
        if row_max_found:
            row_max=row_max-i
            break

    for i in range(s[1]):
        for j in range(s[0]):
            if x[j,i]>0:
                col_min_found=True
        if col_min_found:
            col_min=i
            break
    for i in range(s[1]):
        for j in range(s[0]):
            if x[j,col_max-i]>0:
                col_max_found=True
        if col_max_found:
            col_max=col_max-i
            break
    flag=1
    if row_min==0 and col_min==0:
        flag=0

    return [row_min,col_min,row_max-row_min,col_max-col_min,flag]


def generate_video(loc,out_loc,size=None,fc=2,draw_lines=(5,11)):
    writer=skvideo.io.FFmpegWriter(out_loc,verbosity=1)
    # start the FFmpeg writing subprocess with following parameters
#    writer = skvideo.io.FFmpegWriter(out_loc, outputdict={
#      '-vcodec': 'libx264', '-b': '300000000'
#    })
    print('VIDEO WRITER HAS BEEN RESTORED. PROCEED NORMALLY.')
    f=hf.File(loc,'r')
    r=f['T1'].shape[1]
    c=f['T2'].shape[2]
    print(f['T1'].shape)
    if size is None:
        print ('Setting NUm of pics:',f['T2'].shape)
        size = (f['T2'].shape)[0]
    img=np.zeros(shape=(r*2,c*3))

    for i in range(size):
        img[0:r,0:c]=f['T1'][i,:,:,0]
        img[0:r,c:c*2]=f['T2'][i,:,:,0]
        img[0:r,c*2:c*3]=255*f['label'][i,:,:,0]
        img[r:2*r,0:c]=255*f['label'][i,:,:,1]
        img[r:2*r,c:c*2]=255*f['label'][i,:,:,2]
        img[r:2*r,c*2:c*3]=255*f['label'][i,:,:,3]
        cv.putText(img,str(i),(int(2.1*c),int(0.2*r) ),cv.FONT_HERSHEY_SIMPLEX,1,25)
        img=draw_gridlines(img,2*r,3*c,draw_lines)
        for i1 in range(fc):
            writer.writeFrame(img)
    f.close()
    writer.close()
    
#%%
def generate_video_of_label_volume(label_volume,out_loc,fc=5,draw_lines=(5,5)):
    writer=skvideo.io.FFmpegWriter(out_loc,verbosity=1)
    assert(label_volume.shape==(256,256,256,4))
    r=256
    c=256
    img=np.zeros(shape=(r*2,c*2))

    for i in range(256):
        img[0:r,0:c]=255*label_volume[i,:,:,0]
        img[0:r,c:c*2]=255*label_volume[i,:,:,1]
        img[r:r*2,0:c]=255*label_volume[i,:,:,2]
        img[r:2*r,c:c*2]=255*label_volume[i,:,:,3]
        cv.putText(img,str(i),(int(0.2*c),int(0.2*r) ),cv.FONT_HERSHEY_SIMPLEX,1,100)
        img=draw_gridlines(img,2*r,2*c,draw_lines)
        for i1 in range(fc):
            writer.writeFrame(img)
    writer.close()
#%%

def draw_gridlines(img,r,c,draw_lines,color=100):
    divr=draw_lines[0]+1
    divc=draw_lines[1]+1
    for j in range(draw_lines[0]):
        cv.line(img,(0,int((j+1)*r/divr)),(c,int((j+1)*r/divr)),color,1+j%2,cv.LINE_AA)
    for k in range(draw_lines[1]):
        cv.line(img,(int((k+1)*c/divc),0),(int((k+1)*c/divc),r),color,1+k%2,cv.LINE_AA)
    return img


def generate_chunked_video(loc,out_loc,fc=2,chunks=[],box=(52,203,57,208)):
    writer=skvideo.io.FFmpegWriter(out_loc,verbosity=1)
    f=hf.File(loc,'r')
    r=(box[1]-box[0]+1)
    c=(box[3]-box[2]+1)
    img=np.zeros(shape=(r*2,c*3))
    for items in chunks:
        size=items[2]-items[1]+1
        k=0
        for j in range(size):
            i=items[1]+j
            img[0:r,0:c]=f['T1'][i,box[0]:box[1]+1,box[2]:box[3]+1,0]
            img[0:r,c:c*2]=f['T2'][i,box[0]:box[1]+1,box[2]:box[3]+1,0]
            img[0:r,c*2:c*3]=255*f['label'][i,box[0]:box[1]+1,box[2]:box[3]+1,0]
            img[r:2*r,0:c]=255*f['label'][i,box[0]:box[1]+1,box[2]:box[3]+1,1]
            img[r:2*r,c:c*2]=255*f['label'][i,box[0]:box[1]+1,box[2]:box[3]+1,2]
            img[r:2*r,c*2:c*3]=255*f['label'][i,box[0]:box[1]+1,box[2]:box[3]+1,3]
            cv.putText(img,str(i)+":"+str(k),(int(2.2*c),int(0.2*r)),cv.FONT_HERSHEY_SIMPLEX,2,25)
            k+=1
            for i in range(fc):
                writer.writeFrame(img)
    f.close()
    writer.close()

def generate_cropped_dataset(loc,out_loc,total_frames,chunks=[],box=(52,203,57,208)):
    g=hf.File(out_loc,'w')
    f=hf.File(loc,'r')
    r=(box[1]-box[0]+1)
    c=(box[3]-box[2]+1)
    g.create_dataset('T1',shape=(total_frames,r,c,1),dtype='float32')
    g.create_dataset('T2',shape=(total_frames,r,c,1),dtype='float32')
    g.create_dataset('label',shape=(total_frames,r,c,4),dtype='float32')
    k=0
    subject=1
    for items in chunks:
        size=items[2]-items[1]+1
        print("="*10,'sub:',subject,'size',size,'='*10)
        for j in range(size):
            i=items[1]+j
            g['T1'][k]=f['T1'][i,box[0]:box[1]+1,box[2]:box[3]+1,:]
            g['T2'][k]=f['T2'][i,box[0]:box[1]+1,box[2]:box[3]+1,:]
            g['label'][k]=f['label'][i,box[0]:box[1]+1,box[2]:box[3]+1,:]
            k+=1
            print(k,"<-",i)
        subject+=1
    f.close()
    g.close()

def generate_video_only_inputs(loc,out_loc,size=2560,fc=2,draw_lines=(5,11)):
    writer=skvideo.io.FFmpegWriter(out_loc,verbosity=1)
    f=hf.File(loc,'r')
    r=f['T1'].shape[1]
    c=f['T2'].shape[2]
    print(f['T1'].shape)
    img=np.zeros(shape=(r*1,c*2))

    for i in range(size):
        img[0:r,0:c]=f['T1'][i,:,:,0]
        img[0:r,c:c*2]=f['T2'][i,:,:,0]
        cv.putText(img,str(i),(int(1.1*c),int(0.2*r) ),cv.FONT_HERSHEY_SIMPLEX,1,255)
        img=draw_gridlines(img,r,2*c,draw_lines)
        for i1 in range(fc):
            writer.writeFrame(img)
    f.close()
    writer.close()


def generate_video_enhanced_inputs_CI(loc,out_loc,from_=0,size=256,fc=2,draw_lines=(5,11)):
    writer=skvideo.io.FFmpegWriter(out_loc,verbosity=1)
    f=hf.File(loc,'r')
    r=f['T1'].shape[1]
    c=f['T2'].shape[2]
    print(f['T1'].shape)
    img=np.zeros(shape=(r*1,c*3))

    for i in range(size):
        T1w=f['T1'][(from_+i),:,:,0]
        T2w=f['T2'][(from_+i),:,:,0]
        s=0
        for j in range(200):
            s=0.01*j
            img[0:r,0:c]=T1w
            img[0:r,c:c*2]=T2w
            img[0:r,c*2:c*3]=np.clip((T1w*(T1w-s*T2w)/(T1w+s*T2w)),0,255 ).astype('uint8')
            cv.putText(img,str(i)+"("+str(s)+")",(int(1.1*c),int(0.2*r) ),cv.FONT_HERSHEY_SIMPLEX,1,255)
            img=draw_gridlines(img,r,3*c,draw_lines)
            for i1 in range(fc):
                writer.writeFrame(img)
    f.close()
    writer.close()
    
def generate_video_enhanced_inputs_CLAHE(loc,out_loc,from_=0,size=256,fc=15,draw_lines=(11,11)):
    writer=skvideo.io.FFmpegWriter(out_loc,verbosity=1)
    f=hf.File(loc,'r')
    r=f['T1'].shape[1]
    c=f['T2'].shape[2]
    print(f['T1'].shape)
    img=np.zeros(shape=(r*2,c*2))

    for i in range(size):
        T1w=(f['T1'][(from_+i),:,:,0]).astype('uint8')
        T2w=(f['T2'][(from_+i),:,:,0]).astype('uint8')
        s=0
        for j in range(20):
            s=1+j
            
            # create a CLAHE object (Arguments are optional).
            for k in range(10):
                c_lim=0.2*(k+1)
                img[0:r,0:c]=T1w
                img[0:r,c:c*2]=T2w
                #c_lim=2.0
                clahe = cv.createCLAHE(clipLimit=c_lim, tileGridSize=(s,s))
                cl1 = clahe.apply(T1w)
                cl2 = clahe.apply(T2w)
                img[r:2*r,0:c]=cl1
                img[r:2*r,c:c*2]=cl2
                cv.putText(img,str(from_+i)+"GS:"+str(s)+"CL:"+str(c_lim),(int(0.1*c),int(0.2*r) ),cv.FONT_HERSHEY_SIMPLEX,1,255)
                img=draw_gridlines(img,r*2,2*c,draw_lines)
                for i1 in range(fc):
                    writer.writeFrame(img)
    f.close()
    writer.close()


def generate_all_videos (subjects,axes):
    """sample name: subject-1-axis-2.h5""" 
    
    inp = TRAIN_FOLDER+os.sep+'subject-'
    op = TRAIN_FOLDER+os.sep+'/AXIS_'+str(axes[0])+'/subject-'
    inp = op
    for i in subjects:
        for j in axes:
            fn = inp+str(i)+'-axis-'+str(j)+'.h5'
            op_fn =  op+str(i)+'-axis-'+str(j)+'.mp4'
            generate_video(fn,op_fn,None,8)
            
def generate_dataset_with_less_bg (subjects,axis,bps):
    """sample name: subject-1-axis-2.h5""" 
    assert(len(subjects)==len(bps))
    inp = TRAIN_FOLDER+os.sep+'subject-'
    op = TRAIN_FOLDER+os.sep+'LESS_BG/subject-'
    
    for i in subjects:
        fn = inp+str(i)+'-axis-'+str(axis)+'.h5'
        op_fn =  op+str(i)+'-axis-'+str(axis)+'_lbg.h5'
        j=0
        for k in range(len(bps)):
            if i==bps[j][0]:
                break
            j+=1
        num = bps[j][2]-bps[j][1]+1
               
        print('At j=',j,'bp',bps[j],'and i=',i)
        generate_cropped_dataset(fn,op_fn,
                             num,[bps[j]],box=(0,255,0,255) )
            
            
            
if __name__ == '__main__':
    #%%Using Combined Image
    #generate_video_enhanced_inputs_CI(TRAIN_FOLDER+"/train_set_with_less_bg.h5",OUTPUT_FOLDER+
    # "/enhanceT11_axial.mp4",from_ =163,size=2,fc=11)
    
    #%%Using CLAHE(Contrast Limited Adaptive Histogram Equalization)
    #generate_video_enhanced_inputs_CLAHE(TRAIN_FOLDER+"/axial_with_less_bg.h5",OUTPUT_FOLDER+
    # "/enhanceT1_CLAHE_axial_CL.mp4",from_ =163,size=2,fc=11)
#%% 
    print('About to generate video.\n')
#    name='/coronal_with_less_bg_train.h5'
#    generate_video(TRAIN_FOLDER+name,TRAIN_FOLDER+name[:-3]+'.mp4',size=256,fc=8)

#%%FOR create videos of all subjects
    subjects = [1,2,3,4,5,6,7,8,9,10]
    axes=[2,]
    generate_all_videos(subjects,axes)
    
    #%%INFO ABOUT DATASET
    break_points = [(1,65,171), (2,322,421), (3,574,677), (4,835,927), (5,1084,1188),
                  (6,1348,1447), (7,1604,1703), (8,1854,1954), (9,2105,2206),
                  (10,2365,2462)]
    bp=[(1,65,171),(2,319,425),(3,573,679),(4,828,934),(5,1083,1189),
        (6,1345,1451),(7,1601,1707),(8,1851,1957),(9,2102,2208),(10,2361,2467)]
    
    cor_bps=[(1,61,207),(2,317,458),(3,579,713),(4,836,965),(5,1081,1226),
             (6,1339,1479),(7,1595,1734),(8,1852,1993),(9,2111,2248),(10,2365,2510)]
    
    cor_bp=[(1,61,207),(2,314,460),(3,573,719),(4,827,973),(5,1080,1226),
            (6,1336,1482),(7,1592,1738),(8,1849,1995),(9,2106,2252),(10,2364,2510)]
    
    bps_axis1 = [(1,60,207),(2,60,202),(3,66,201),(4,67,197),(5,56,202),
                 (6,55,199),(7,54,198),(8,59,201),(9,62,200),(10,60,206)]
    
    bps_axis2 =  [ (1,84,191), (2,90,190), (3,89,194), (4,96,189), (5,91,196), (6,88,188),
                  (7,88,188), (8,93,194), (9,97,199), (10,97,195) ]
    
    bps_axis0 = [(1,64,187), (2,69,187),(3,66,187),(4,68,189),(5,65,188),
                 (6,70,184), (7,68,185), (8,67,185), (9,64,192), (10,68,184)]
#creating dataset with less bg
#    generate_dataset_with_less_bg(subjects,0,bps_axis0)
    
    #generate_chunked_video(OUTPUT_FOLDER+"/train_set.h5",OUTPUT_FOLDER+"/axial_cropped.mp4",fc=11,
    #                      chunks=bp,size=256)
    #
    i=1
    for items in cor_bp:
       print(i,items[0],items[1],":",items[2]-items[1])
    
    #generate_cropped_dataset(OUTPUT_FOLDER+"/train_set_corona.h5",OUTPUT_FOLDER+"/train_set_coronal_with_less_bg.h5",
    #                         147*10,cor_bp,box=(0,255,0,255) )
    #


