#!/usr/bin/env python3

import numpy as np
import glob,os,sys, ipdb
import argparse, tempfile
from deep_util_JW import *
sys.path.append(os.path.dirname(__file__))

parser = argparse.ArgumentParser('   ==========   Fetal U_Net segmentation script made by Jinwoo Hong (04.15.2020 ver.2)   ==========   ')
parser.add_argument('-input', '--input_MR',action='store',dest='inp',type=str, required=True, help='input MR file name (\'.nii or .nii.gz\') or folder name')
parser.add_argument('-output', '--output_loc',action='store',dest='out',type=str, required=True, help='Output path')
parser.add_argument('-axi', '--axi_weight',action='store',dest='axi',default=os.path.dirname(os.path.abspath(__file__))+'/axi.h5',type=str, help='Axial weight file')
parser.add_argument('-cor', '--cor_weight',action='store',dest='cor',default=os.path.dirname(os.path.abspath(__file__))+'/cor.h5',type=str, help='Coronal weight file')
parser.add_argument('-sag', '--sag_weight',action='store',dest='sag',default=os.path.dirname(os.path.abspath(__file__))+'/sag.h5',type=str, help='Sagittal weight file')
args = parser.parse_args()

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

if os.path.isdir(args.inp):
    img_list = np.asarray(sorted(glob.glob(args.inp+'/*.nii*')))
elif os.path.isfile(args.inp):
    img_list = np.asarray(sorted(glob.glob(args.inp)))
else:
    img_list = np.asarray(sorted(glob.glob(args.inp)))


if len(img_list)==0:
    print('No such file or dictectory')
    exit()

mask= os.path.dirname(os.path.abspath(__file__))+'/mask31_D10.nii.gz'



with tempfile.TemporaryDirectory() as tmpdir:
    test_dic, _ =make_dic(img_list, img_list, mask, 'axi', 0)
    model = Unet_network([128,128,1], 5,ite=3, depth=4).build()
    model.load_weights(args.axi)

    tmask = model.predict(test_dic)
    make_result(tmask,img_list,mask,tmpdir+'/','axi')
    tmask = model.predict(test_dic[:,::-1,:,:])
    make_result(tmask[:,::-1,:,:],img_list,mask,tmpdir+'/','axi','f1')
    tmask = model.predict(axfliper(test_dic))
    make_result(axfliper(tmask,1),img_list,mask,tmpdir+'/','axi','f2')
    tmask = model.predict(axfliper(test_dic[:,::-1,:,:]))
    make_result(axfliper(tmask[:,::-1,:,:],1),img_list,mask,tmpdir+'/','axi','f3')

    del model, tmask, test_dic
    reset_graph()

    test_dic, _ =make_dic(img_list, img_list, mask, 'cor', 0)
    model = Unet_network([128,128,1], 5,ite=3, depth=4).build()
    model.load_weights(args.cor)

    tmask = model.predict(test_dic)
    make_result(tmask,img_list,mask,tmpdir+'/','cor')
    tmask = model.predict(test_dic[:,:,::-1,:])
    make_result(tmask[:,:,::-1,:],img_list,mask,tmpdir+'/','cor','f1')
    tmask = model.predict(cofliper(test_dic))
    make_result(cofliper(tmask,1),img_list,mask,tmpdir+'/','cor','f2')
    tmask = model.predict(cofliper(test_dic[:,:,::-1,:]))
    make_result(cofliper(tmask[:,:,::-1,:],1),img_list,mask,tmpdir+'/','cor','f3')

    del model, tmask, test_dic
    reset_graph()


    test_dic, _ =make_dic(img_list, img_list, mask, 'sag', 0)
    model = Unet_network([128,128,1], 3, ite=3, depth=4).build()
    model.load_weights(args.sag)

    tmask = model.predict(test_dic)
    make_result(tmask,img_list,mask,tmpdir+'/','sag')
    tmask = model.predict(test_dic[:,::-1,:,:])
    make_result(tmask[:,::-1,:,:],img_list,mask,tmpdir+'/','sag','f1')
    tmask = model.predict(test_dic[:,:,::-1,:])
    make_result(tmask[:,:,::-1,:],img_list,mask,tmpdir+'/','sag','f2')

    del model, tmask, test_dic
    reset_graph()

    if np.shape(img_list):
        for i2 in range(len(img_list)): 
            filename=img_list[i2].split('/')[-1:][0]
            filename=filename.split('.nii')[0]
            make_sum(tmpdir+'/'+filename+'*axi*', tmpdir+'/'+filename+'*cor*',tmpdir+'/'+filename+'*sag*', img_list[i2], args.out+'/')
            make_verify(img_list[i2], args.out+'/')
    else:
        make_sum(tmpdir+'/'+filename+'*axi*', tmpdir+'/'+filename+'*cor*',tmpdir+'/'+filename+'*sag*', img_list, args.out+'/')
        make_verify(img_list, args.out+'/')
