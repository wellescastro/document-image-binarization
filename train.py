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
from util.EarlyStopping import EarlyStopping
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

# np.set_printoptions(threshold=np.nan)

def criterion(logits, labels):
    return losses.FBeta_ScoreLoss().forward(logits, labels)
    # return losses.F1ScoreLoss().forward(logits, labels)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight.data) # kernel_initializer
        nn.init.zeros_(m.bias.data) # bias initializer

def main():
    # Hyperparameters
    batch_size = 10
    epochs = 200
    threshold = 0.5
    early_stopping_patience = 20
    window_size = 256,256
    strides = 96,96

    # Training informartion
    model_weiths_path = "checkpoints/"
    resume_training = True
    if resume_training is False:
        start_epoch = 0
    else:
        start_epoch = 11
    model_name = "dibco2016_256x256_whole_aug"
    resume_checkpoint = model_weiths_path + "{}-epoch-{}.pth".format(model_name, start_epoch)

    use_cuda = torch.cuda.is_available()

    # define the model and optimizer
    net = AutoEncoder(nb_layers=5).cuda()
    net.apply(weights_init)
    print(net)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, verbose=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.75, last_epoch=-1)

    # define the dataset and data augmentation operations
    training_transforms = transforms.Compose([
                transforms.ToTensor()
                ])

    training_target_transforms = transforms.Compose([
                transforms.ToTensor()
                ])


    training_set = DIBCODataset(years=[2009, 2010, 2011, 2012, 2013, 2014],
    transform = training_transforms, target_transform=training_target_transforms, window_size=window_size, stride=strides, include_augmentation=True
    )

    testing_transforms = transforms.Compose([ 
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

    # # optionally resume from a checkpoint
    if resume_training:
        start_epoch, early_stopper.best, early_stopper.num_bad_epochs = load_checkpoint(net, optimizer, scheduler, resume_checkpoint, verbose=True)

    cudnn.benchmark = True

    for epoch in range(start_epoch, epochs):
        training_metrics = {'loss':0, 'mse':0, 'f1score':0, 'time': 0}
        testing_metrics = {'loss':0, 'mse':0, 'f1score':0, 'time': 0}

        # scheduler.step() # enable learning rate decay every 30 epochs

        # perform training
        net.train()
        for ind, (inputs, target) in enumerate(train_loader):
            
            t0 = time.time()    
            if use_cuda:
                inputs = inputs.cuda()
                target = target.cuda()

            inputs, target = Variable(inputs), Variable(target)

            # forward
            logits = net.forward(inputs)
            loss = criterion(logits.view(-1, 256*256), target.view(-1, 256*256))

            # backward + optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # torch.nn.utils.clip_grad_norm_(net.parameters(), 0.25) # perform gradient clipping

            training_metrics['loss'] += loss.item()
            # training_metrics['mse'] += mse_score(logits.data, target)
            training_metrics['time'] += (time.time() - t0)

            # # get thresholded prediction and compute the f1-score per patches
            training_metrics['f1score'] += f_score_no_threshold(target.view(-1, 256*256), logits.view(-1, 256*256), threshold=threshold).item()
    
        # get the average training loss
        training_metrics['loss'] /= len(train_loader)
        training_metrics['mse'] /= len(train_loader)
        training_metrics['f1score'] /= len(train_loader)
        
        # scheduler.step(training_metrics['loss']) # enable reduce learning rate on plateau

        # get the average of the metrics
        testing_metrics['loss'] /= len(test_loader)
        testing_metrics['mse'] /= len(test_loader)
        testing_metrics['f1score'] /= len(test_loader)

        test_f1score = final_evaluation(net, testing_set, testing_transforms, window_size, strides, threshold)
        print('Epoch %d/%d train_loss: %.4f train_f1score: %.4f current patience: %d avg train time: %.2fs test fscore: %.4f' %
                    (epoch + 1, epochs, training_metrics['loss'], training_metrics['f1score'], 
                    (early_stopper.patience - early_stopper.num_bad_epochs), training_metrics['time'], test_f1score))

        
        if early_stopper.step(training_metrics['loss']) == True:
            print("Early stopping triggered!")
            break

        # save checkpoint
        is_best = training_metrics['loss'] == early_stopper.best
 
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'best_training_loss': early_stopper.best,
            'num_bad_epochs': early_stopper.num_bad_epochs,
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict(),
        }, is_best, model_weiths_path, model_name)

        torch.cuda.empty_cache()
    

    load_weights(net, "checkpoints/model_best.pth", verbose=True)

    # starting final evaluation using the reconstructed images
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

                idxs = list(np.nonzero(patch_predicted))
                if len(idxs[0]) > 0:
                    idxs[0] += coords[0]
                    idxs[1] += coords[2]
                    count_strides[idxs] += 1

                # this is an alternative code to perform the same operation
                # for i in range(patch_predicted.shape[0]):
                #     for j in range(patch_predicted.shape[1]):
                #         if patch_predicted[i, j] > 0:
                #             count_strides[coords[0]+i,coords[2]+j] += 1
                
                img_prediction[coords[0]:coords[1],coords[2]:coords[3]] = img_prediction[coords[0]:coords[1],coords[2]:coords[3]] + prediction_unpadded + 1e-30
                
            img_prediction /= count_strides # disable if do not work
            img_prediction = img_prediction > 0.5
            img_prediction = 255 * img_prediction
            img_prediction = 255 - img_prediction
            img_prediction = img_prediction.astype(np.uint8)
            
            fmeasure, pfmeasure, psnr, nrm, mpm, drd = compute_metrics(img_prediction.copy(), img_gt.copy())
            fmeasures.append(fmeasure)

            # sk_fmeasures.append(sk_metrics.fbeta_score( ((255 - img_gt.ravel()) / 255), ((255 - img_prediction.ravel()) / 255), 1))
        
        return np.mean(fmeasures)
        

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
                i_f = img.shape[0] - 1
            
            if j_f > img.shape[1]:
                j_f = img.shape[1] - 1
            
            yield (i_0, i_f, j_0, j_f), img[i_0:i_f, j_0:j_f]


if __name__ == '__main__':
    # TODO: add args functionality
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--network",type=str,default="fusionnet",help="choose between fusionnet & unet")
    # parser.add_argument("--batch_size",type=int,default=1,help="batch size")
    # parser.add_argument("--num_gpu",type=int,default=1,help="number of gpus")
    # args = parser.parse_args()

    main()
