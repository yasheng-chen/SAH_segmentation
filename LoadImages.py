    
import SimpleITK as sitk
from multiprocessing import Pool
import numpy as np
import math
import os

class ScanFile(object):   
    def __init__(self,directory,prefix=None,postfix=None):  
        self.directory=directory  
        self.prefix=prefix  
        self.postfix=postfix  
          
    def scan_files(self):    
        files_list=[]    
            
        for dirpath,dirnames,filenames in os.walk(self.directory):   
            ''''' 
            dirpath is a string, the path to the directory.   
            dirnames is a list of the names of the subdirectories in dirpath (excluding '.' and '..'). 
            filenames is a list of the names of the non-directory files in dirpath. 
            '''  
            for special_file in filenames:    
                if self.postfix:  
                    if  special_file.endswith(self.postfix):    
                        files_list.append(os.path.join(dirpath,special_file))    
                elif self.prefix:    
                    if special_file.startswith(self.prefix):  
                        files_list.append(os.path.join(dirpath,special_file))    
                else:    
                    files_list.append(os.path.join(dirpath,special_file))    
                                  
        return files_list    
      
    def scan_subdir(self):  
        subdir_list=[]  
        for dirpath,dirnames,files in os.walk(self.directory):  
            subdir_list.append(dirpath)  
        return subdir_list      


#########################################################################################################
def extract_mask_slices(mknp):
  
    [dimz,dimx,dimy]=mknp.shape
    #print('extract_mask_slices dims %d %d %d'%(dimz,dimx,dimy))

    sl_list = []
   
    for i in range(dimz):
        if (np.amax(mknp[i,:,:]) > 0):
            sl_list.append(i)

    return sl_list


########################################################################################################
def load_image_single(ct_fn, mk_fn):
    ct = sitk.ReadImage(ct_fn)
    ctnp = sitk.GetArrayFromImage(ct)

    mk = sitk.ReadImage(mk_fn)
    mknp = sitk.GetArrayFromImage(mk)

    return ctnp, mknp  

########################################################################################################
def load_images(path_images):
    scan_images = ScanFile(path_images, postfix = 'normalized.nii.gz')  
    filenames_images = scan_images.scan_files()  
      
    ct_list = []
    mk_list = []
    id_list = []
    sl_list = []

    ###############################################################
    index_subject = 0

    for filename in filenames_images:         
        print('train: ', filename)
        
        ct_fn = filename
        mk_fn = filename.replace('normalized.nii.gz','seg.nii.gz')
	        
        ct = sitk.ReadImage(ct_fn)
        ctnp = sitk.GetArrayFromImage(ct)

        mk = sitk.ReadImage(mk_fn)
        mknp = sitk.GetArrayFromImage(mk)
  
        ct_list.append(ctnp)
        mk_list.append(mknp)

        index_slices = extract_mask_slices(mknp)

        for k in range(len(index_slices)):
            id_list.append(index_subject)
            sl_list.append(index_slices[k])

        index_subject = index_subject+1

    return ct_list, mk_list, id_list, sl_list

#######################################################################################

def extract_batch(ct_list, mk_list, id_list, sl_list, batch_size, index_interest):
   
    num_samples = len(id_list)

    [dimz, dimx, dimy] = ct_list[0].shape
    
    #ct_tmp=np.zeros([batch_size, 3, dimx, dimy], dtype=np.float32)
    ct_tmp=np.zeros([batch_size, 1, dimx, dimy], dtype=np.float32)
    mk_tmp=np.zeros([batch_size, 1, dimx, dimy], dtype=np.float32)

    ind = np.random.choice(num_samples, size=batch_size, replace=False)

    for i in range(batch_size):
        iii = id_list[ind[i]]
        sss = sl_list[ind[i]]

        #ct_tmp[i,0,:,:] = ct_list[iii][sss-1, :, :]
        #ct_tmp[i,1,:,:] = ct_list[iii][sss  , :, :]
        #ct_tmp[i,2,:,:] = ct_list[iii][sss+1, :, :]

        ct_tmp[i,0,:,:] = ct_list[iii][sss, :, :]
        mk_tmp[i,0,:,:] = mk_list[iii][sss, :, :]

    mk_tmp[np.where(mk_tmp!=index_interest)] = 0
    mk_tmp[np.where(mk_tmp==index_interest)] = 1	    

    ct_tmp, mk_tmp = BSpline2D.apply_bspline_deform(ct_tmp, mk_tmp, 32)

    return ct_tmp, mk_tmp 

#######################################################################################

def extract_batch_single(ct, mk):
   
    [dimz, dimx, dimy] = mk.shape

    index_slices = extract_mask_slices(mk)

    num_slices = len(index_slices)

    ct_tmp=np.zeros([num_slices, 1, dimx, dimy], dtype=np.float32)

    for i in range(num_slices):
        sss = index_slices[i]
        ct_tmp[i,0,:,:] = ct[sss, :, :]

        #ct_tmp[i,0,:,:] = ct[sss-1, :, :]
        #ct_tmp[i,1,:,:] = ct[sss  , :, :]
        #ct_tmp[i,2,:,:] = ct[sss+1, :, :]

    return ct_tmp, index_slices 
