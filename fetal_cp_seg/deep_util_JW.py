class Unet_network:
    def __init__(self, input_shape, out_ch, loss='dice_loss', metrics=['dice_coef', 'dis_dice_coef'], ite=2, depth=4, dim=32, weights='', init='he_normal',acti='elu',lr=1e-4, dropout=0):
        from keras.layers import Input
        self.input_shape=input_shape
        self.out_ch=out_ch
        self.loss = loss
        self.metrics = metrics
        self.ite=ite
        self.depth=depth
        self.dim=dim
        self.init=init
        self.acti=acti
        self.weight=weights
        self.dropout=dropout
        self.lr=lr
        self.I = Input(input_shape)

    def conv_block(self,inp,dim):
        from keras.layers import BatchNormalization as bn, Activation, Conv2D, Dropout
        x = bn()(inp)
        x = Activation(self.acti)(x)
        x = Conv2D(dim, (3,3), padding='same', kernel_initializer=self.init)(x)
        return x

    def conv1_block(self,inp,dim):
        from keras.layers import BatchNormalization as bn, Activation, Conv2D, Dropout
        x = bn()(inp)
        x = Activation(self.acti)(x)
        x = Conv2D(dim, (1,1), padding='same', kernel_initializer=self.init)(x)
        return x

    def tconv_block(self,inp,dim):
        from keras.layers import BatchNormalization as bn, Activation, Conv2DTranspose, Dropout
        x = bn()(inp)
        x = Activation(self.acti)(x)
        x = Conv2DTranspose(dim, 2, strides=2, padding='same', kernel_initializer=self.init)(x)
        return x

    def basic_block(self, inp, dim):
        for i in range(self.ite):
            inp = self.conv_block(inp,dim)
        return inp

    def build_U(self, inp, dim, depth):
        from keras.layers import MaxPooling2D, concatenate, Dropout
        if depth > 0:
            x = self.basic_block(inp, dim)
            x2 = MaxPooling2D()(x)
            if (self.dropout>0) & (depth<4): x2 = Dropout(self.dropout)(x2)
            x2 = self.build_U(x2, int(dim*2), depth-1)
            x2 = self.tconv_block(x2,int(dim*2))
            x2 = concatenate([x,x2])
            x2 = self.basic_block(x2, dim)
            if (self.dropout>0) & (depth<4): x2 = Dropout(self.dropout)(x2)
        else:
            x2 = self.basic_block(inp, dim)
        return x2

    def UNet(self):
        from keras.layers import Conv2D
        from keras.models import Model
        from keras.optimizers import Adam
        o = self.build_U(self.I, self.dim, self.depth)
        o = Conv2D(self.out_ch, 1, activation='softmax')(o)
        model = Model(inputs=self.I, outputs=o)
        if len(self.metrics)==2:
            model.compile(optimizer=Adam(lr=self.lr), loss=getattr(self, self.loss), metrics=[getattr(self, self.metrics[0]),getattr(self, self.metrics[1])])
        else:
            model.compile(optimizer=Adam(lr=self.lr), loss=getattr(self, self.loss), metrics=[getattr(self, self.metrics[0])])
        if self.weight:
            model.load_weights(self.weight)
        return model

    def build(self):
        return self.UNet()

    def dice_coef(self, y_true, y_pred):
        from keras import backend as K
        smooth = 0.001
        intersection = K.sum(y_true * K.round(y_pred), axis=[1,2])
        union = K.sum(y_true, axis=[1,2]) + K.sum(K.round(y_pred), axis=[1,2])
        dice = K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
        return K.mean(dice[1:])

    def dice_loss(self, y_true, y_pred):
        from keras import backend as K
        smooth = 0.001
        intersection = K.sum(y_true * y_pred, axis=[1,2])
        union = K.sum(y_true, axis=[1,2]) + K.sum(y_pred, axis=[1,2])
        dice = K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
        return K.mean(K.pow(-K.log(dice[1:]),0.3))

    def dis_loss(self, y_true, y_pred):
        from keras import backend as K
        import cv2, numpy as np
        si=K.int_shape(y_pred)[-1]
        riter=3
        smooth = 0.001
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(riter*2+1,riter*2+1))
        kernel=kernel/(np.sum(kernel))
        kernel=np.repeat(kernel[:,:,np.newaxis],si,axis=-1)
        kernel=K.variable(kernel[:,:,:,np.newaxis])
        y_true_s=K.depthwise_conv2d(y_true,kernel,data_format="channels_last",padding="same")
        y_pred_s=K.depthwise_conv2d(y_pred,kernel,data_format="channels_last",padding="same")
        y_true_s = y_true_s > 0.8
        y_pred_s = y_pred_s > 0.8
        y_true_s = y_true - K.cast(y_true_s,'float32')
        y_pred_s = y_pred - K.cast(y_pred_s,'float32')
        intersection = K.sum(y_true_s * y_pred_s, axis=[1,2])
        union = K.sum(y_true_s, axis=[1,2]) + K.sum(y_pred_s, axis=[1,2])
        dice = K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
        return K.sum(K.pow(-K.log(dice[1:]),0.3))

    def dis_dice_coef(self, y_true, y_pred):
        from keras import backend as K
        import cv2, numpy as np
        si=K.int_shape(y_pred)[-1]
        riter=3
        smooth = 0.001
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(riter*2+1,riter*2+1))
        kernel=kernel/(np.sum(kernel))
        kernel=np.repeat(kernel[:,:,np.newaxis],si,axis=-1)
        kernel=K.variable(kernel[:,:,:,np.newaxis])
        y_true_s=K.depthwise_conv2d(y_true,kernel,data_format="channels_last",padding="same")
        y_pred_s=K.depthwise_conv2d(y_pred,kernel,data_format="channels_last",padding="same")
        y_true_s = y_true_s > 0.8
        y_pred_s = y_pred_s > 0.8
        y_true_s = y_true - K.cast(y_true_s,'float32')
        y_pred_s = y_pred - K.cast(y_pred_s,'float32')
        intersection = K.sum(y_true_s * y_pred_s, axis=[1,2])
        union = K.sum(y_true_s, axis=[1,2]) + K.sum(y_pred_s, axis=[1,2])
        dice = K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
        return K.mean(dice[1:])

    def hyb_loss(self, y_true, y_pred):
        d_loss=self.dice_loss(y_true, y_pred)
        h_loss=self.dis_loss(y_true, y_pred)
        return 0.1*h_loss + d_loss


def reset_graph():
    from keras import backend as K
    import tensorflow as tf
    K.clear_session()
    tf.reset_default_graph()


def axfliper(array,f=0):
    import numpy as np
    if f:
        array = array[:,:,::-1,:]
        array2 = np.concatenate((array[:,:,:,0,np.newaxis],array[:,:,:,2,np.newaxis],array[:,:,:,1,np.newaxis],
                                 array[:,:,:,4,np.newaxis],array[:,:,:,3,np.newaxis]),axis=-1)
        return array2
    else:
        array = array[:,:,::-1,:]
    return array

def cofliper(array,f=0):
    import numpy as np
    if f:
        array = array[:,::-1,:,:]
        array2 = np.concatenate((array[:,:,:,0,np.newaxis],array[:,:,:,2,np.newaxis],array[:,:,:,1,np.newaxis],
                                 array[:,:,:,4,np.newaxis],array[:,:,:,3,np.newaxis]),axis=-1)
        return array2
    else:
        array = array[:,::-1,:,:]
    return array

def make_dic(img_list, gold_list, mask, dim,flip=0):
    import numpy as np
    import nibabel as nib
    def get_data(img, label, mask_im, mask_min, mask_max, repre_size):
        import nibabel as nib
        img = np.squeeze(nib.load(img).get_data()) * mask_im
        img = img[mask_min[0]:mask_max[0],mask_min[1]:mask_max[1],mask_min[2]:mask_max[2]]
        loc = np.where(img<np.percentile(img,2))
        img[loc]=0
        loc = np.where(img>np.percentile(img,98))
        img[loc]=0
        loc = np.where(img)
        img[loc] = (img[loc] - np.mean(img[loc])) / np.std(img[loc])
        label = np.squeeze(nib.load(label).get_data())
        label = label[mask_min[0]:mask_max[0],mask_min[1]:mask_max[1],mask_min[2]:mask_max[2]]
        return img, label

    mask_im = nib.load(mask).get_data()
    mask_min = np.min(np.where(mask_im),axis=1)
    mask_max = np.max(np.where(mask_im),axis=1)
    repre_size = mask_max-mask_min
#     max_repre = np.max(repre_size)
    max_repre = 128
    if dim == 'axi':
        dic = np.zeros([repre_size[2]*len(img_list), max_repre, max_repre,1])
        seg = np.zeros([repre_size[2]*len(img_list), max_repre, max_repre,5])
    elif dim == 'cor':
        dic = np.zeros([repre_size[1]*len(img_list), max_repre, max_repre,1])
        seg = np.zeros([repre_size[1]*len(img_list), max_repre, max_repre,5])
    elif dim == 'sag':
        dic = np.zeros([repre_size[0]*len(img_list), max_repre, max_repre,1])
        seg = np.zeros([repre_size[0]*len(img_list), max_repre, max_repre,3])
    else:
        print('available: axi, cor, sag.   Your: '+dim)
        exit()

    for i in range(0, len(img_list)):
        img, label = get_data(img_list[i], gold_list[i], mask_im, mask_min, mask_max, repre_size)
        if dim == 'axi':
            img2 = np.pad(img,((int((max_repre-img.shape[0])/2),int((max_repre-img.shape[0])/2)),
                               (int((max_repre-img.shape[1])/2),int((max_repre-img.shape[1])/2)),
                               (0,0)), 'constant')
            dic[repre_size[2]*i:repre_size[2]*(i+1),:,:,0]= np.swapaxes(img2,2,0)
            img2 = np.pad(label,((int((max_repre-img.shape[0])/2),int((max_repre-img.shape[0])/2)),
                                 (int((max_repre-img.shape[1])/2),int((max_repre-img.shape[1])/2)),
                                 (0,0)), 'constant')
            img2 = np.swapaxes(img2,2,0)
        elif dim == 'cor':
            img2 = np.pad(img,((int((max_repre-img.shape[0])/2),int((max_repre-img.shape[0])/2)),
                               (0,0),
                               (int((max_repre-img.shape[2])/2),int((max_repre-img.shape[2])/2))),'constant')
            dic[repre_size[1]*i:repre_size[1]*(i+1),:,:,0]= np.swapaxes(img2,1,0)
            img2 = np.pad(label,((int((max_repre-img.shape[0])/2),int((max_repre-img.shape[0])/2)),
                                 (0,0),
                                 (int((max_repre-img.shape[2])/2),int((max_repre-img.shape[2])/2))),'constant')
            img2 = np.swapaxes(img2,1,0)
        elif dim == 'sag':
            img2 = np.pad(img,((0,0),
                              (int((max_repre-img.shape[1])/2),int((max_repre-img.shape[1])/2)),
                              (int((max_repre-img.shape[2])/2),int((max_repre-img.shape[2])/2))), 'constant')
            dic[repre_size[0]*i:repre_size[0]*(i+1),:,:,0]= img2
            img2 = np.pad(label,((0,0),
                                (int((max_repre-img.shape[1])/2),int((max_repre-img.shape[1])/2)),
                                (int((max_repre-img.shape[2])/2),int((max_repre-img.shape[2])/2)),), 'constant')
        else:
            print('available: axi, cor, sag.   Your: '+dim)
            exit()
        if (dim == 'axi') | (dim == 'cor'):
            img3 = np.zeros_like(img2)
            back_loc = np.where(img2<0.5)
            left_in_loc = np.where((img2>160.5)&(img2<161.5))
            right_in_loc = np.where((img2>159.5)&(img2<160.5))
            left_plate_loc = np.where((img2>0.5)&(img2<1.5))
            right_plate_loc = np.where((img2>41.5)&(img2<42.5))
            img3[back_loc]=1
        elif dim == 'sag':
            img3 = np.zeros_like(img2)
            back_loc = np.where(img<0.5)
            in_loc = np.where((img2>160.5)&(img2<161.5)|(img2>159.5)&(img2<160.5))
            plate_loc = np.where((img2>0.5)&(img2<1.5)|(img2>41.5)&(img2<42.5))
            img3[back_loc]=1
        else:
            print('available: axi, cor, sag.   Your: '+dim)
            exit()

        if dim == 'axi':
            seg[repre_size[2]*i:repre_size[2]*(i+1),:,:,0]=img3
            img3[:]=0
            img3[left_in_loc]=1
            seg[repre_size[2]*i:repre_size[2]*(i+1),:,:,1]=img3
            img3[:]=0
            img3[right_in_loc]=1
            seg[repre_size[2]*i:repre_size[2]*(i+1),:,:,2]=img3
            img3[:]=0
            img3[left_plate_loc]=1
            seg[repre_size[2]*i:repre_size[2]*(i+1),:,:,3]=img3
            img3[:]=0
            img3[right_plate_loc]=1
            seg[repre_size[2]*i:repre_size[2]*(i+1),:,:,4]=img3
            img3[:]=0
        elif dim == 'cor':
            seg[repre_size[1]*i:repre_size[1]*(i+1),:,:,0]=img3
            img3[:]=0
            img3[left_in_loc]=1
            seg[repre_size[1]*i:repre_size[1]*(i+1),:,:,1]=img3
            img3[:]=0
            img3[right_in_loc]=1
            seg[repre_size[1]*i:repre_size[1]*(i+1),:,:,2]=img3
            img3[:]=0
            img3[left_plate_loc]=1
            seg[repre_size[1]*i:repre_size[1]*(i+1),:,:,3]=img3
            img3[:]=0
            img3[right_plate_loc]=1
            seg[repre_size[1]*i:repre_size[1]*(i+1),:,:,4]=img3
            img3[:]=0
        elif dim == 'sag':
            seg[repre_size[0]*i:repre_size[0]*(i+1),:,:,0]=img3
            img3[:]=0
            img3[in_loc]=1
            seg[repre_size[0]*i:repre_size[0]*(i+1),:,:,1]=img3
            img3[:]=0
            img3[plate_loc]=1
            seg[repre_size[0]*i:repre_size[0]*(i+1),:,:,2]=img3
            img3[:]=0
        else:
            print('available: axi, cor, sag.   Your: '+dim)
            exit()
    if flip:
        if dim == 'axi':
            dic=np.concatenate((dic,dic[:,::-1,:,:]),axis=0)
            seg=np.concatenate((seg,seg[:,::-1,:,:]),axis=0)
            dic=np.concatenate((dic, axfliper(dic)),axis=0)
            seg=np.concatenate((seg, axfliper(seg, 1)),axis=0)
        elif dim == 'cor':
            dic=np.concatenate((dic,dic[:,:,::-1,:]),axis=0)
            seg=np.concatenate((seg,seg[:,:,::-1,:]),axis=0)
            dic=np.concatenate((dic, cofliper(dic)),axis=0)
            seg=np.concatenate((seg, cofliper(seg, 1)),axis=0)
        elif dim == 'sag':
            dic = np.concatenate((dic, dic[:,:,::-1,:], dic[:,::-1,:,:]),axis=0)
            seg = np.concatenate((seg, seg[:,:,::-1,:], seg[:,::-1,:,:]),axis=0)
        else:
            print('available: axi, cor, sag.   Your: '+dim)
            exit()
    return dic, seg


def make_result(tmask, img_list, mask,result_loc,axis,ext=''):
    import nibabel as nib
    import numpy as np
    mask_im = nib.load(mask).get_data()
    mask_min = np.min(np.where(mask_im),axis=1)
    mask_max = np.max(np.where(mask_im),axis=1)
    repre_size = mask_max-mask_min
#     max_repre = np.max(repre_size)
    max_repre = 128

    if np.shape(img_list):
        for i2 in range(len(img_list)):
            print('filename : '+img_list[i2])
            img = nib.load(img_list[i2])
            img_data = np.squeeze(img.get_data())
            img2=img_data[mask_min[0]:mask_max[0],mask_min[1]:mask_max[1],mask_min[2]:mask_max[2]]
            pr4=tmask[i2*(np.int(tmask.shape[0]/len(img_list))):(i2+1)*(np.int(tmask.shape[0]/len(img_list)))]
            if axis == 'axi':
                pr4=np.swapaxes(np.argmax(pr4,axis=3).astype(np.int),0,2)
                pr4=pr4[int((max_repre-img2.shape[0])/2):-int((max_repre-img2.shape[0])/2),:,:]
            elif axis == 'cor':
                pr4=np.swapaxes(np.argmax(pr4,axis=3).astype(np.int),0,1)
                pr4=pr4[int((max_repre-img2.shape[0])/2):-int((max_repre-img2.shape[0])/2),:,int((max_repre-img2.shape[2])/2):-int((max_repre-img2.shape[2])/2)]
            elif axis == 'sag':
                pr4=np.argmax(pr4,axis=3).astype(np.int)
                pr4=pr4[:,:,int((max_repre-img2.shape[2])/2):-int((max_repre-img2.shape[2])/2)]
            else:
                print('available: axi, cor, sag.   Your: '+axis)
                exit()

            img_data[:] = 0
            img_data[mask_min[0]:mask_max[0],mask_min[1]:mask_max[1],mask_min[2]:mask_max[2]]=pr4
            new_img = nib.Nifti1Image(img_data, img.affine, img.header)
            filename=img_list[i2].split('/')[-1:][0]
            filename=filename.split('.nii')[0]
            if axis== 'axi':
                filename=filename+'_deep_axi'+ext+'.nii.gz'
            elif axis== 'cor':
                filename=filename+'_deep_cor'+ext+'.nii.gz'
            elif axis== 'sag':
                filename=filename+'_deep_sag'+ext+'.nii.gz'
            else:
                print('available: axi, cor, sag.   Your: '+axis)
                exit()
            print('save result : '+result_loc+filename)
            nib.save(new_img, result_loc+str(filename))
    else:
        print('filename : '+img_list)
        img = nib.load(img_list)
        img_data = np.squeeze(img.get_data())
        img2=img_data[mask_min[0]:mask_max[0],mask_min[1]:mask_max[1],mask_min[2]:mask_max[2]]
        pr4 = tmask
        if axis == 'axi':
            pr4=np.swapaxes(np.argmax(pr4,axis=3).astype(np.int),0,2)
            pr4=pr4[int((max_repre-img2.shape[0])/2):-int((max_repre-img2.shape[0])/2),:,:]
        elif axis == 'cor':
            pr4=np.swapaxes(np.argmax(pr4,axis=3).astype(np.int),0,1)
            pr4=pr4[int((max_repre-img2.shape[0])/2):-int((max_repre-img2.shape[0])/2),:,int((max_repre-img2.shape[2])/2):-int((max_repre-img2.shape[2])/2)]
        elif axis == 'sag':
            pr4=np.argmax(pr4,axis=3).astype(np.int)
            pr4=pr4[:,:,int((max_repre-img2.shape[2])/2):-int((max_repre-img2.shape[2])/2)]
        else:
            print('available: axi, cor, sag.   Your: '+axis)
            exit()

        img_data[:] = 0
        img_data[mask_min[0]:mask_max[0],mask_min[1]:mask_max[1],mask_min[2]:mask_max[2]]=pr4
        new_img = nib.Nifti1Image(img_data, img.affine, img.header)
        filename=img_list.split('/')[-1:][0]
        filename=filename.split('.nii')[0]
        if axis== 'axi':
            filename=filename+'_deep_axi'+ext+'.nii.gz'
        elif axis== 'cor':
            filename=filename+'_deep_cor'+ext+'.nii.gz'
        elif axis== 'sag':
            filename=filename+'_deep_sag'+ext+'.nii.gz'
        else:
            print('available: axi, cor, sag.   Your: '+axis)
            exit()

        print('save result : '+result_loc+filename)
        nib.save(new_img, result_loc+str(filename))

    return 1


def make_sum(axi_filter, cor_filter, sag_filter, input_name, result_loc):
    import nibabel as nib
    import numpy as np
    import sys, glob

    # 1-->axi 2-->cor 3-->sag
    axi_list = sorted(glob.glob(axi_filter))
    cor_list = sorted(glob.glob(cor_filter))
    sag_list = sorted(glob.glob(sag_filter))
    axi = nib.load(axi_list[0])
    cor = nib.load(cor_list[0])
    sag = nib.load(sag_list[0])

    bak = np.zeros(np.shape(axi.get_data()))
    left_in = np.zeros(np.shape(axi.get_data()))
    right_in = np.zeros(np.shape(axi.get_data()))
    left_plate = np.zeros(np.shape(axi.get_data()))
    right_plate = np.zeros(np.shape(axi.get_data()))
    total = np.zeros(np.shape(axi.get_data()))

    for i in range(len(axi_list)):
        axi_data = nib.load(axi_list[i]).get_data()
        cor_data = nib.load(cor_list[i]).get_data()
        if len(sag_list) > i:
            sag_data = nib.load(sag_list[i]).get_data()

        loc = np.where(axi_data==0)
        bak[loc]=bak[loc]+1
        loc = np.where(cor_data==0)
        bak[loc]=bak[loc]+1
        if len(sag_list) > i:
            loc = np.where(sag_data==0)
            bak[loc]=bak[loc]+1

        loc = np.where(axi_data==1)
        left_in[loc]=left_in[loc]+1
        loc = np.where(cor_data==1)
        left_in[loc]=left_in[loc]+1
        if len(sag_list) > i:
            loc = np.where(sag_data==1)
            left_in[loc]=left_in[loc]+1

        loc = np.where(axi_data==2)
        right_in[loc]=right_in[loc]+1
        loc = np.where(cor_data==2)
        right_in[loc]=right_in[loc]+1
        if len(sag_list) > i:
            loc = np.where(sag_data==1)
            right_in[loc]=right_in[loc]+1

        loc = np.where(axi_data==3)
        left_plate[loc]=left_plate[loc]+1
        loc = np.where(cor_data==3)
        left_plate[loc]=left_plate[loc]+1
        if len(sag_list) > i:
            loc = np.where(sag_data==2)
            left_plate[loc]=left_plate[loc]+1

        loc = np.where(axi_data==4)
        right_plate[loc]=right_plate[loc]+1
        loc = np.where(cor_data==4)
        right_plate[loc]=right_plate[loc]+1
        if len(sag_list) > i:
            loc = np.where(sag_data==2)
            right_plate[loc]=right_plate[loc]+1

    result = np.concatenate((bak[np.newaxis,:], left_in[np.newaxis,:], right_in[np.newaxis,:], left_plate[np.newaxis,:], right_plate[np.newaxis,:]),axis=0)
    result = np.argmax(result, axis=0)
    #relabel
    ori_label = np.array([1,2,3,4])
    relabel = np.array([161,160,1,42])
    for itr in range(len(ori_label)):
        loc = np.where((result>ori_label[itr]-0.5)&(result<ori_label[itr]+0.5))
        result[loc]=relabel[itr]
    filename=input_name.split('/')[-1:][0]
    filename=filename.split('.nii')[0]
    filename=filename+'_deep_agg.nii.gz'
    new_img = nib.Nifti1Image(result, axi.affine, axi.header)
    nib.save(new_img, result_loc+'/'+filename)
    print('Aggregation finishied!')
    print('save file : '+result_loc+'/'+filename)

def make_verify(img_list, result_loc):
    import numpy as np
    import nibabel as nib
    import matplotlib.pyplot as plt
    import sys
    print('Verify image making...', end=" ")
    if np.shape(img_list):
        for i2 in range(len(img_list)):
            img = nib.load(img_list[i2]).get_data()
            label_name = img_list[i2].split('/')[-1].split('.nii')[0]
            label_name = label_name+'_deep_agg.nii.gz'
            label = nib.load(result_loc+'/'+label_name).get_data()
            #ori_label = np.array([1,2,3,4])
            #relabel = np.array([161,160,1,42])
            #for itr in range(len(relabel)):
            #    loc = np.where((label>relabel[itr]-0.5)&(label<relabel[itr]+0.5))
            #    label[loc]=ori_label[itr]
            f,axarr = plt.subplots(3,3,figsize=(9,9))
            f.patch.set_facecolor('k')

            f.text(0.4, 0.95, label_name, size="large", color="White")

            axarr[0,0].imshow(np.rot90(img[:,:,np.int(img.shape[-1]*0.4)]),cmap='gray')
            axarr[0,0].imshow(np.rot90(label[:,:,np.int(label.shape[-1]*0.4)]),alpha=0.3, cmap='gnuplot2')
            axarr[0,0].axis('off')

            axarr[0,1].imshow(np.rot90(img[:,:,np.int(img.shape[-1]*0.5)]),cmap='gray')
            axarr[0,1].imshow(np.rot90(label[:,:,np.int(label.shape[-1]*0.5)]),alpha=0.3, cmap='gnuplot2')
            axarr[0,1].axis('off')

            axarr[0,2].imshow(np.rot90(img[:,:,np.int(img.shape[-1]*0.6)]),cmap='gray')
            axarr[0,2].imshow(np.rot90(label[:,:,np.int(label.shape[-1]*0.6)]),alpha=0.3, cmap='gnuplot2')
            axarr[0,2].axis('off')

            axarr[1,0].imshow(np.rot90(img[:,np.int(img.shape[-2]*0.4),:]),cmap='gray')
            axarr[1,0].imshow(np.rot90(label[:,np.int(label.shape[-2]*0.4),:]),alpha=0.3, cmap='gnuplot2')
            axarr[1,0].axis('off')

            axarr[1,1].imshow(np.rot90(img[:,np.int(img.shape[-2]*0.5),:]),cmap='gray')
            axarr[1,1].imshow(np.rot90(label[:,np.int(label.shape[-2]*0.5),:]),alpha=0.3, cmap='gnuplot2')
            axarr[1,1].axis('off')

            axarr[1,2].imshow(np.rot90(img[:,np.int(img.shape[-2]*0.6),:]),cmap='gray')
            axarr[1,2].imshow(np.rot90(label[:,np.int(label.shape[-2]*0.6),:]),alpha=0.3, cmap='gnuplot2')
            axarr[1,2].axis('off')

            axarr[2,0].imshow(np.rot90(img[np.int(img.shape[0]*0.4),:,:]),cmap='gray')
            axarr[2,0].imshow(np.rot90(label[np.int(label.shape[0]*0.4),:,:]),alpha=0.3, cmap='gnuplot2')
            axarr[2,0].axis('off')

            axarr[2,1].imshow(np.rot90(img[np.int(img.shape[0]*0.5),:,:]),cmap='gray')
            axarr[2,1].imshow(np.rot90(label[np.int(label.shape[0]*0.5),:,:]),alpha=0.3, cmap='gnuplot2')
            axarr[2,1].axis('off')

            axarr[2,2].imshow(np.rot90(img[np.int(img.shape[0]*0.6),:,:]),cmap='gray')
            axarr[2,2].imshow(np.rot90(label[np.int(label.shape[0]*0.6),:,:]),alpha=0.3, cmap='gnuplot2')
            axarr[2,2].axis('off')
            f.subplots_adjust(wspace=0, hspace=0)
            plt.savefig(result_loc+'/'+label_name.split('/')[-1].split('.nii')[0]+'_verify.png', facecolor=f.get_facecolor())
    else:
        img = nib.load(img_list).get_data()
        label_name = img_list.split('/')[-1].split('.nii')[0]
        label_name = label_name+'_deep_agg.nii.gz'
        label = nib.load(result_loc+'/'+label_name).get_data()
        #ori_label = np.array([1,2,3,4])
        #relabel = np.array([161,160,1,42])
        #for itr in range(len(relabel)):
        #    loc = np.where((label>relabel[itr]-0.5)&(label<relabel[itr]+0.5))
        #    label[loc]=ori_label[itr]

        f,axarr = plt.subplots(3,3,figsize=(9,9))
        f.patch.set_facecolor('k')

        f.text(0.4, 0.95, label_name, size="large", color="White")

        axarr[0,0].imshow(np.rot90(img[:,:,np.int(img.shape[-1]*0.4)]),cmap='gray')
        axarr[0,0].imshow(np.rot90(label[:,:,np.int(label.shape[-1]*0.4)]),alpha=0.3, cmap='gnuplot2')
        axarr[0,0].axis('off')

        axarr[0,1].imshow(np.rot90(img[:,:,np.int(img.shape[-1]*0.5)]),cmap='gray')
        axarr[0,1].imshow(np.rot90(label[:,:,np.int(label.shape[-1]*0.5)]),alpha=0.3, cmap='gnuplot2')
        axarr[0,1].axis('off')

        axarr[0,2].imshow(np.rot90(img[:,:,np.int(img.shape[-1]*0.6)]),cmap='gray')
        axarr[0,2].imshow(np.rot90(label[:,:,np.int(label.shape[-1]*0.6)]),alpha=0.3, cmap='gnuplot2')
        axarr[0,2].axis('off')

        axarr[1,0].imshow(np.rot90(img[:,np.int(img.shape[-2]*0.4),:]),cmap='gray')
        axarr[1,0].imshow(np.rot90(label[:,np.int(label.shape[-2]*0.4),:]),alpha=0.3, cmap='gnuplot2')
        axarr[1,0].axis('off')

        axarr[1,1].imshow(np.rot90(img[:,np.int(img.shape[-2]*0.5),:]),cmap='gray')
        axarr[1,1].imshow(np.rot90(label[:,np.int(label.shape[-2]*0.5),:]),alpha=0.3, cmap='gnuplot2')
        axarr[1,1].axis('off')

        axarr[1,2].imshow(np.rot90(img[:,np.int(img.shape[-2]*0.6),:]),cmap='gray')
        axarr[1,2].imshow(np.rot90(label[:,np.int(label.shape[-2]*0.6),:]),alpha=0.3, cmap='gnuplot2')
        axarr[1,2].axis('off')

        axarr[2,0].imshow(np.rot90(img[np.int(img.shape[0]*0.4),:,:]),cmap='gray')
        axarr[2,0].imshow(np.rot90(label[np.int(label.shape[0]*0.4),:,:]),alpha=0.3, cmap='gnuplot2')
        axarr[2,0].axis('off')

        axarr[2,1].imshow(np.rot90(img[np.int(img.shape[0]*0.5),:,:]),cmap='gray')
        axarr[2,1].imshow(np.rot90(label[np.int(label.shape[0]*0.5),:,:]),alpha=0.3, cmap='gnuplot2')
        axarr[2,1].axis('off')

        axarr[2,2].imshow(np.rot90(img[np.int(img.shape[0]*0.6),:,:]),cmap='gray')
        axarr[2,2].imshow(np.rot90(label[np.int(label.shape[0]*0.6),:,:]),alpha=0.3, cmap='gnuplot2')
        axarr[2,2].axis('off')
        f.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(result_loc+'/'+label_name.split('/')[-1].split('.nii')[0]+'_verify.png', facecolor=f.get_facecolor())
    print('Done!')
    return 0

def make_callbacks(weight_name, history_name, monitor='val_loss', patience=100, mode='min', save_weights_only=True):
    from keras.callbacks import Callback
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    import six, io, time, csv, numpy as np, json, warnings
    from collections import deque
    from collections import OrderedDict
    from collections import Iterable
    from collections import defaultdict
    from keras.utils.generic_utils import Progbar
    from keras import backend as K
    from keras.engine.training_utils import standardize_input_data
    class CSVLogger_time(Callback):
        """Callback that streams epoch results to a csv file.
        Supports all values that can be represented as a string,
        including 1D iterables such as np.ndarray.
        # Example
        ```python
        csv_logger = CSVLogger('training.log')
        model.fit(X_train, Y_train, callbacks=[csv_logger])
        ```
        # Arguments
            filename: filename of the csv file, e.g. 'run/log.csv'.
            separator: string used to separate elements in the csv file.
            append: True: append if file exists (useful for continuing
                training). False: overwrite existing file,
        """

        def __init__(self, filename, separator=',', append=False):
            self.sep = separator
            self.filename = filename
            self.append = append
            self.writer = None
            self.keys = None
            self.append_header = True
            if six.PY2:
                self.file_flags = 'b'
                self._open_args = {}
            else:
                self.file_flags = ''
                self._open_args = {'newline': '\n'}
            super(CSVLogger_time, self).__init__()

        def on_train_begin(self, logs=None):
            if self.append:
                if os.path.exists(self.filename):
                    with open(self.filename, 'r' + self.file_flags) as f:
                        self.append_header = not bool(len(f.readline()))
                mode = 'a'
            else:
                mode = 'w'
            self.csv_file = io.open(self.filename,
                                    mode + self.file_flags,
                                    **self._open_args)

        def on_epoch_begin(self, epoch, logs={}):
            self.epoch_time_start = time.time()

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}

            def handle_value(k):
                is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
                if isinstance(k, six.string_types):
                    return k
                elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                    return '"[%s]"' % (', '.join(map(str, k)))
                else:
                    return k

            if self.keys is None:
                self.keys = sorted(logs.keys())

            if self.model.stop_training:
                # We set NA so that csv parsers do not fail for this last epoch.
                logs = dict([(k, logs[k] if k in logs else 'NA') for k in self.keys])

            if not self.writer:
                class CustomDialect(csv.excel):
                    delimiter = self.sep
                fieldnames = ['epoch'] + self.keys +['time']
                if six.PY2:
                    fieldnames = [unicode(x) for x in fieldnames]
                self.writer = csv.DictWriter(self.csv_file,
                                             fieldnames=fieldnames,
                                             dialect=CustomDialect)
                if self.append_header:
                    self.writer.writeheader()

            row_dict = OrderedDict({'epoch': epoch})
            logs['time']=time.time() - self.epoch_time_start
            self.keys.append('time')
            row_dict.update((key, handle_value(logs[key])) for key in self.keys)
            self.writer.writerow(row_dict)
            self.csv_file.flush()

        def on_train_end(self, logs=None):
            self.csv_file.close()
            self.writer = None

        def __del__(self):
            if hasattr(self, 'csv_file') and not self.csv_file.closed:
                self.csv_file.close()
    earlystop=EarlyStopping(monitor=monitor, patience=patience, verbose=0, mode=mode)
    checkpoint=ModelCheckpoint(filepath=weight_name, monitor=monitor, mode=mode, save_best_only=True, save_weights_only=save_weights_only, verbose=0)
    csvlog=CSVLogger_time(history_name, separator='\t')
    return [earlystop, checkpoint, csvlog]



