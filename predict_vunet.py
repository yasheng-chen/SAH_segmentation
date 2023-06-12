#CUDA_VISIBLE_DEVICES=3 python train.py --cuda --outpath ./outputs
from __future__ import print_function
import argparse
import os
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
from vunet import unet_512
from LoadImages import ScanFile, load_image_single, extract_batch_single
import SimpleITK as sitk

# Training settings
parser = argparse.ArgumentParser(description='Example')
parser.add_argument('--ndf', type=int, default=16, help='Number of root features, Default=16')
parser.add_argument('--normalization', type=int, default=0, help='0 NoNorm 1 BatchNorm 2 InstanceNorm, Default=1')
parser.add_argument('--imagepath', default='/tmp', help='folder to load images')
parser.add_argument('--modelname', default='/tmp', help='folder to load model')
parser.add_argument('--outpath', default='/tmp', help='folder to output images')

opt = parser.parse_args()

print(opt)

try:
    os.makedirs(opt.outpath)
except OSError:
    pass

print('===> Building model')

prob_dropout=0

NetS = unet_512(1, 1, opt.ndf, prob_dropout, opt.normalization, 1)

###############################################
image_inputs = ScanFile(opt.imagepath, postfix = 'normalized.nii.gz')
filenames = image_inputs.scan_files()
num_files = len(filenames)

NetS.load_state_dict(torch.load(opt.modelname))
NetS.eval()

for filename in filenames:
    ct_fn = filename
    mk_fn = filename.replace('normalized.nii.gz', 'seg.nii.gz')

    sg_out_fn = filename.replace('normalized.nii.gz', 'seg.nii.gz')
    p, f = os.path.split(sg_out_fn)
    sg_out_fn = opt.outpath + '/' + f

    ct_out_fn = filename.replace('normalized.nii.gz', 'rec.nii.gz')
    p, f = os.path.split(ct_out_fn)
    ct_out_fn = opt.outpath + '/' + f

    print("output file %s"%(sg_out_fn))
    print("ct_fn %s mk_fn %s"%(ct_fn, mk_fn))

    ct, mk = load_image_single(ct_fn, mk_fn)    
    ct_batch, index_slices = extract_batch_single(ct, mk)
    [dimz, dimx, dimy] = ct.shape

    ct_itk = sitk.ReadImage(ct_fn)
    spacing   = ct_itk.GetSpacing()
    origin    = ct_itk.GetOrigin()
    direction = ct_itk.GetDirection()

    num_batches = ct_batch.shape[0]
    sg_out = np.zeros([dimz, dimx, dimy])
    ct_out = np.zeros([dimz, dimx, dimy])

    for indexi in range(num_batches):

        ct_tmp = np.zeros([1, 1, dimx, dimy])

        ct_tmp[0,:,:,:] = ct_batch[indexi,:,:,:].astype(float)

        ct_tmp = torch.from_numpy(ct_tmp)
        ct_tmp = ct_tmp.type(torch.FloatTensor)

        ct_tmp = Variable(ct_tmp)

        output_seg = NetS(ct_tmp)

        output_seg = output_seg.cpu().detach().numpy()
        sg_out[index_slices[indexi],:,:] = output_seg[0,:,:]

    #########################################
    sg_out.astype(np.float)
    sg_out = sitk.GetImageFromArray(sg_out)
    sg_out.SetSpacing(spacing)
    sg_out.SetOrigin(origin)
    sg_out.SetDirection(direction)
    sitk.WriteImage(sg_out, sg_out_fn)
