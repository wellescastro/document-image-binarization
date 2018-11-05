import numpy as np
import os
import cv2
import math
import torch
import torch.optim as optim
from torch.autograd import Variable
from sklearn import metrics as sk_metrics
from torch.utils.data import DataLoader
import torch.nn as nn
from util.DIBCOReader import DIBCODataset
from torchvision import transforms
from models.AutoEncoder import AutoEncoder
from util import losses
from EarlyStopping import EarlyStopping
from util.metrics import f_score, mse_score, f_score_no_threshold
from util.utils import save_checkpoint, load_checkpoint, load_weights
from skimage.util.shape import view_as_blocks
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from util.sliding_window import sliding_window_view
from util.dbico_metrics import compute_metrics
from skimage.transform import warp, AffineTransform
import time
import torch.backends.cudnn as cudnn
from loss import LossBinary

def criterion(logits, labels):
    # return losses.FBeta_ScoreLoss().forward(logits, labels)
    return losses.F1ScoreLoss().forward(logits, labels)

class ToTensorNoScale(object):
    def __init__(self):
        pass
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            if isinstance(img, torch.ByteTensor):
                return img.float()
            else:
                return img

        # handle PIL Image
        img = torch.from_numpy(np.array(pic, np.uint8, copy=False))
        
        nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        
        if isinstance(img, torch.ByteTensor):
            return img.float()
        return img

def main():
    # Hyperparameters
    batch_size = 25
    epochs = 300
    threshold = 0.5
    early_stopping_patience = 20
    window_size = 256,256
    strides = 128,128

    # Training informartion
    model_weiths_path = "checkpoints/"
    resume_training = False
    if resume_training is False:
        start_epoch = 0
    else:
        start_epoch = 10
    model_name = "auto_encoder_aug5"
    resume_checkpoint = model_weiths_path + "{}-epoch-{}.pth".format(model_name, start_epoch)

    use_cuda = torch.cuda.is_available()

    # define the model and optimizer
    net = AutoEncoder(nb_layers=5).cuda()
    print(net)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
  
    # define the dataset and data augmentation operations
    training_transforms = transforms.Compose([
                # transforms.ToPILImage(mode='L'),
                # # transforms.RandomResizedCrop(window_size[0], scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
                transforms.RandomAffine(degrees=0, scale=(0.5,1.5)),
                # # RandomAffineTransform(scale_range=(0.5,1.5),rotation_range=(0,0),shear_range=(0,0), translation_range=(0,0)),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                # ToTensorNoScale(),
                transforms.ToTensor()
                # transforms.Normalize(mean=[0.5], std=[0.5])
                ])

    training_target_transforms = transforms.Compose([
                # transforms.ToPILImage(mode='L'),
                # # transforms.RandomResizedCrop(window_size[0], scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
                transforms.RandomAffine(degrees=0, scale=(0.5,1.5)),
                # # RandomAffineTransform(scale_range=(0.5,1.5),rotation_range=(0,0),shear_range=(0,0), translation_range=(0,0)),
                # transforms.RandomHorizontalFlip(), 
                # transforms.RandomVerticalFlip(),
                # # ToTensorNoScale(),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5], std=[0.5])
                ])


    training_set = DIBCODataset(years=[2009, 2010, 2011, 2012, 2013, 2014],
    transform = training_transforms, target_transform=training_target_transforms, window_size=window_size, stride=strides, include_augmentation=False
    )

    testing_transforms = transforms.Compose([ 
        # ToTensorNoScale(),
        transforms.ToTensor()
        ])

    testing_target_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    testing_set = DIBCODataset(years=[2016], 
    transform = testing_transforms, target_transform=testing_target_transforms, window_size=window_size, stride=strides, include_augmentation=False
    )

    train_loader = DataLoader(training_set,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True # CUDA only
                         )
    
    test_loader = DataLoader(testing_set,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=4,
                          pin_memory=True # CUDA only
                         )

    print("Training on {} samples and testing on {} samples "
          .format(len(train_loader.dataset), len(test_loader.dataset)))

    early_stopper = EarlyStopping(mode='min', patience=early_stopping_patience)

    # optionally resume from a checkpoint
    if resume_training:
        start_epoch, early_stopper.best, early_stopper.num_bad_epochs = load_checkpoint(net, optimizer, resume_checkpoint)
    cudnn.benchmark = True

    load_weights(net, "checkpoints/model_best.pth")

    print(final_evaluation(net, testing_set, testing_transforms, window_size, strides, threshold))

def final_evaluation(net, testing_set, testing_transforms, window_size, strides, threshold):
    fmeasures = []
    sk_fmeasures = []
    scaler = MinMaxScaler(feature_range=(0,255))
    net.eval()
    with torch.no_grad():
        for filename_gr in testing_set.data_files:
            filename_gt = filename_gr.replace("GR", "GT")
                            
            img_gr = cv2.imread(filename_gr, cv2.IMREAD_GRAYSCALE)
            img_gt = cv2.imread(filename_gt, cv2.IMREAD_GRAYSCALE)

            #preprocessing
            img_gr = 255 - img_gr

            img_prediction = np.zeros(img_gr.shape, dtype=np.uint8)
            for coords, patch in sliding_window(img_gr, strides, window_size):
                patch_size = patch.shape

                vertical_padding = (window_size[0] - patch_size[0])
                horizontal_padding = (window_size[1] - patch_size[1])
                patch_gr = np.pad(patch, ((0,vertical_padding),(0,horizontal_padding)),mode='constant')
                patch_gr_with_channel = np.expand_dims(patch_gr, 2) # change to 0 if using final transforms
    
                inputs = testing_transforms(patch_gr_with_channel).cuda()
                inputs = inputs.unsqueeze(1)

                # forward
                logits = net(inputs)
                pred = (logits > 0.5).float() # threshold the output activation map
                pred = logits.squeeze() # remove all dimensions of size 1 (batch dimension and channel dimension)
                
                prediction_unpadded = pred[0:window_size[0]-vertical_padding, 0:window_size[1]-horizontal_padding]
                img_prediction[coords[0]:coords[1],coords[2]:coords[3]] = prediction_unpadded
                
            img_prediction = 255 * img_prediction
            img_prediction = 255 - img_prediction
            
            fmeasure, pfmeasure, psnr, nrm, mpm, drd = compute_metrics(img_prediction.copy(), img_gt.copy())
            fmeasures.append(fmeasure)

            sk_fmeasures.append(sk_metrics.fbeta_score( ((255 - img_gt.ravel()) / 255), ((255 - img_prediction.ravel()) / 255), 1))
        
        return np.mean(fmeasures)
        

def sliding_window(img, strides, window_size):
    shape = img.shape
    tiles_per_row = int( np.ceil( shape[0] / float(strides[0]) ) )
    tiles_per_col = int( np.ceil( shape[1] / float(strides[1]) ) )

    for i in range(tiles_per_row):
        for j in range(tiles_per_col):
            i_0 = strides[0]*i
            i_f = i_0 + window_size[0]

            j_0 = strides[1]*j
            j_f = j_0 + window_size[1]
            
            yield (i_0,i_f, j_0,j_f), img[i_0:i_f, j_0:j_f]


if __name__ == '__main__':
    # TODO: add args functionality
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--network",type=str,default="fusionnet",help="choose between fusionnet & unet")
    # parser.add_argument("--batch_size",type=int,default=1,help="batch size")
    # parser.add_argument("--num_gpu",type=int,default=1,help="number of gpus")
    # args = parser.parse_args()

    main()
