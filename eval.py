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
import argparse

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

def main(model_filename, year, save_dir):
    # Hyperparameters
    threshold = 0.5
    window_size = 256,256
    strides = 96,96

    if save_dir is not None and os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    # define the model and optimizer
    net = AutoEncoder(nb_layers=5).cuda()
    print("Initializing evaluation of model {}...".format(model_filename))
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
 
    testing_transforms = transforms.Compose([ 
        # ToTensorNoScale(),
        transforms.ToTensor()
        ])

    testing_target_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    testing_set = DIBCODataset(years=[year], 
    transform = testing_transforms, target_transform=testing_target_transforms, window_size=window_size, stride=strides, include_augmentation=False
    )

    load_weights(net, model_filename)
    f1, pf1 = final_evaluation(net, testing_set, testing_transforms, window_size, strides, threshold, save_dir=save_dir)
    print("Results: F1-Score: {:4f} Pseudo F1-Score: {:4f}\n".format(f1, pf1))

def final_evaluation(net, testing_set, testing_transforms, window_size, strides, threshold, save_dir=None):
    fmeasures = []
    pfmeasures = []
    scaler = MinMaxScaler(feature_range=(0,255))
    net.eval()
    with torch.no_grad():
        for filename_gr in testing_set.data_files:
            filename_gt = filename_gr.replace("GR", "GT")
                            
            img_gr = cv2.imread(filename_gr, cv2.IMREAD_GRAYSCALE)
            img_gt = cv2.imread(filename_gt, cv2.IMREAD_GRAYSCALE)

            #preprocessing
            img_gr = 255 - img_gr

            count_strides = np.ones(img_gr.shape, dtype=int)
            img_prediction = np.zeros(img_gr.shape, dtype=np.float64)
            for coords, patch in sliding_window(img_gr, strides, window_size):
                patch_size = patch.shape

                vertical_padding = (window_size[0] - patch_size[0])
                horizontal_padding = (window_size[1] - patch_size[1])
                patch_gr = np.pad(patch, ((0, vertical_padding), (0, horizontal_padding)),mode='constant')
                patch_gr_with_channel = np.expand_dims(patch_gr, 2) # change to 0 if using final transforms
    
                inputs = testing_transforms(patch_gr_with_channel).cuda()
                inputs = inputs.unsqueeze(1)

                # forward
                logits = net(inputs)
                # pred = (logits > 0.5).float() # threshold the output activation map
                logits.float()
                pred = logits.squeeze() # remove all dimensions of size 1 (batch dimension and channel dimension)
                pred = pred.data.cpu().numpy()

                prediction_unpadded = pred[0:window_size[0]-vertical_padding, 0:window_size[1]-horizontal_padding]
                prediction_unpadded = prediction_unpadded.astype(np.float64)
                # print(prediction_unpadded[0,128])

                patch_predicted = img_prediction[coords[0]:coords[1],coords[2]:coords[3]]

                # for i in range(patch_predicted.shape[0]):
                #     for j in range(patch_predicted.shape[1]):
                #         if patch_predicted[i, j] > 0:
                #             count_strides[coords[0]+i,coords[2]+j] += 1

                idxs = list(np.nonzero(patch_predicted))
                if len(idxs[0]) > 0:
                    idxs[0] += coords[0]
                    idxs[1] += coords[2]
                    count_strides[idxs] += 1
                
                img_prediction[coords[0]:coords[1],coords[2]:coords[3]] = img_prediction[coords[0]:coords[1],coords[2]:coords[3]] + prediction_unpadded + 1e-30
                
            img_prediction /= count_strides # disable if do not work
            img_prediction = img_prediction > 0.5
            img_prediction = 255 * img_prediction
            img_prediction = 255 - img_prediction
            img_prediction = img_prediction.astype(np.uint8)

            if save_dir is not None:
                Image.fromarray(img_prediction).save(save_dir + "/" + filename_gr.split("/")[-1])
            
            fmeasure, pfmeasure, psnr, nrm, mpm, drd = compute_metrics(img_prediction.copy(), img_gt.copy())
            fmeasures.append(fmeasure)
            pfmeasures.append(pfmeasure)

        return np.mean(fmeasures), np.mean(pfmeasure)
        

def sliding_window(img, strides, window_size):
    shape = img.shape
    tiles_row_wise = int( np.ceil( shape[0] / float(strides[0]) ) )
    tiles_col_wise = int( np.ceil( shape[1] / float(strides[1]) ) )

    for i in range(tiles_row_wise):
        for j in range(tiles_col_wise):
            i_0 = strides[0]*i
            i_f = i_0 + window_size[0]

            j_0 = strides[1]*j
            j_f = j_0 + window_size[1]

            if i_f > img.shape[0]:
                i_f = img.shape[0]
     
            
            if j_f > img.shape[1]:
                j_f = img.shape[1] 
     
            
            yield (i_0, i_f, j_0, j_f), img[i_0:i_f, j_0:j_f]


if __name__ == '__main__':
    # TODO: add args functionality
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--network",type=str,default="fusionnet",help="choose between fusionnet & unet")
    # parser.add_argument("--batch_size",type=int,default=1,help="batch size")
    # parser.add_argument("--num_gpu",type=int,default=1,help="number of gpus")
    # args = parser.parse_args()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",type=str,help="please specify the model to load")
    parser.add_argument("--year",type=int,help="please specify the dibco year to test")
    parser.add_argument("--save_dir", type=str, help="in case you want save the image results")
    args = parser.parse_args()
    
    model_filename = args.model
    year = args.year
    save_dir = args.save_dir

    main(model_filename, year, save_dir)
