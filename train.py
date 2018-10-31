import numpy as np
import os
import cv2
import math
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from util.DIBCOReader import DIBCODataset
from torchvision import transforms
from models.AutoEncoder import AutoEncoder
from models import losses
from EarlyStopping import EarlyStopping
from metrics import f2_score, mse_score
from util.utils import save_checkpoint, load_checkpoint, load_weights
from skimage.util.shape import view_as_blocks
from PIL import Image
from util.sliding_window import sliding_window_view
from util.dbico_metrics import compute_metrics
from skimage.transform import warp, AffineTransform
import time


def criterion(logits, labels):
    return losses.F1ScoreLoss().forward(logits, labels)
    # return losses.BinaryCrossEntropyLoss2d().forward(logits, labels)
    # return losses.SoftDiceLoss().forward(logits, labels)

class ToTensorNoScale(object):
    def __init__(self):
        pass
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            if isinstance(img, torch.ByteTensor):
                return img.float().div(255)
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
            return img.float().div(255)
        return img

class RandomAffineTransform(object):
    def __init__(self,
                 scale_range,
                 rotation_range,
                 shear_range,
                 translation_range
                 ):
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.shear_range = shear_range
        self.translation_range = translation_range

    def __call__(self, img):
        img_data = np.array(img)
        h, w, n_chan = img_data.shape
        scale_x = np.random.uniform(*self.scale_range)
        scale_y = np.random.uniform(*self.scale_range)
        scale = (scale_x, scale_y)
        # rotation = np.random.uniform(*self.rotation_range)
        # shear = np.random.uniform(*self.shear_range)
        # translation = (
        #     np.random.uniform(*self.translation_range) * w,
        #     np.random.uniform(*self.translation_range) * h
        # )
        # af = AffineTransform(scale=scale, shear=shear, rotation=rotation, translation=translation)
        af = AffineTransform(scale=scale)
        img_data1 = warp(img_data, af.inverse)
        img_data1 = np.squeeze(img_data1, axis=2)
        img1 = Image.fromarray(np.uint8(img_data1 * 255))
        return img1

def main():
    # Hyperparameters
    batch_size = 100
    epochs = 200
    threshold = 0.5
    early_stopping_patience = 20
    window_size = 256,256
    strides = 128,128

    # Training informartion
    start_epoch = 0
    model_weiths_path = "checkpoints/"
    resume_training = False
    model_name = "auto_encoder3"
    resume_checkpoint = model_weiths_path + "{}-epoch-12.pth".format(model_name)

    use_cuda = torch.cuda.is_available()

    # define the model and optimizer
    net = AutoEncoder(nb_layers=3).cuda()
    print(net)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
  
    # define the dataset and data augmentation operations
    training_transforms = transforms.Compose([
                # transforms.ToPILImage(mode='L'),
                # transforms.RandomResizedCrop(window_size[0], scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
                # transforms.RandomAffine(degrees=0, scale=(0.5,1.5)),
                # RandomAffineTransform(scale_range=(0.5,1.5),rotation_range=(0,0),shear_range=(0,0), translation_range=(0,0)),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor() 
                ])

    training_target_transforms = transforms.Compose([
                # transforms.ToPILImage(mode='L'),
                # transforms.RandomResizedCrop(window_size[0], scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
                # transforms.RandomAffine(degrees=0, scale=(0.5,1.5)),
                # RandomAffineTransform(scale_range=(0.5,1.5),rotation_range=(0,0),shear_range=(0,0), translation_range=(0,0)),
                # transforms.RandomHorizontalFlip(), 
                transforms.ToTensor()
                ])


    training_set = DIBCODataset(years=[2009,2010,2011,2012,2013,2014],
    transform = training_transforms, target_transform=training_transforms, window_size=window_size, stride=strides, include_augmentation=True
    )

    testing_transforms = transforms.Compose([ 
        transforms.ToTensor()])

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

    for epoch in range(start_epoch, epochs):
        training_metrics = {'loss':0, 'mse':0, 'f1score':0, 'time': 0}
        testing_metrics = {'loss':0, 'mse':0, 'f1score':0, 'time': 0}

        # perform training
        net.train()
        t0 = time.time()
        for ind, (inputs, target) in enumerate(train_loader):
            
            if use_cuda:
                inputs = inputs.cuda()
                target = target.cuda()

            inputs, target = Variable(inputs), Variable(target)

            # forward
            logits = net.forward(inputs)
            loss = criterion(logits, target)

            # backward + optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_metrics['loss'] += loss.item()
            training_metrics['mse'] += mse_score(logits.data, target)
            training_metrics['time'] += (time.time() - t0)

            # get thresholded prediction and compute the f1-score per patches
            training_metrics['f1score'] += f2_score(target.view(-1, window_size[0] * window_size[1]), logits.view(-1, window_size[0] * window_size[1]), threshold=threshold).item()
            
        # get the average training loss
        training_metrics['loss'] /= len(train_loader)
        training_metrics['mse'] /= len(train_loader)
        training_metrics['f1score'] /= len(train_loader)
        
        # perform validation 
        net.eval()
        t0 = time.time()
        with torch.no_grad():
            for ind, (inputs, target) in enumerate(test_loader):
                if use_cuda:
                    inputs = inputs.cuda()
                    target = target.cuda()

                inputs, target = Variable(inputs), Variable(target)

                # forward
                logits = net.forward(inputs)
                loss = criterion(logits, target)

                testing_metrics['loss'] += loss.item()
                testing_metrics['mse'] += mse_score(logits.data, target)
                testing_metrics['time'] += (time.time() - t0)

                # get thresholded prediction and compute the f1-score per patches
                testing_metrics['f1score'] += f2_score(target.view(-1, window_size[0] * window_size[1]), logits.view(-1, window_size[0] * window_size[1]), threshold=threshold).item()
        
        scheduler.step(training_metrics['loss'])

        # get the average of the metrics
        testing_metrics['loss'] /= len(test_loader)
        testing_metrics['mse'] /= len(test_loader)
        testing_metrics['f1score'] /= len(test_loader)

        print('[%d, %d] train_loss: %.4f test_loss: %.4f train_mse: %.4f test_mse: %.4f train_f1score: %.4f test_f1score: %.4f current patience: %d avg train time: %.2fs avg test time: %.2fs' %
                    (epoch + 1, epochs, training_metrics['loss'], testing_metrics['loss'], training_metrics['mse'], testing_metrics['mse'], training_metrics['f1score'], testing_metrics['f1score'], 
                    (early_stopper.patience - early_stopper.num_bad_epochs), training_metrics['time'], testing_metrics['time']))

        final_evaluation(net, testing_set, testing_transforms, window_size, strides, threshold)


        if early_stopper.step(training_metrics['loss']) == True:
            print("Early stopping triggered!")
            break

        # save checkpoint
        is_best = training_metrics['loss'] < early_stopper.best

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'best_training_loss': early_stopper.best,
            'num_bad_epochs': early_stopper.num_bad_epochs,
            'optimizer' : optimizer.state_dict(),
        }, is_best, model_weiths_path, model_name)
    

    load_weights(net, "checkpoints/model_best.pth")

    # maybe gonna be used for feeding with imgs between 0 and 255
    # final_transforms = transforms.Compose([ 
    #         transforms.Lambda(lambda cv2img:torch.from_numpy(cv2img).float().to('cuda'))])

    # starting final evaluation using the reconstructed image
    final_evaluation(net, testing_set, testing_transforms, window_size, strides, threshold)

def final_evaluation(net, testing_set, testing_transforms, window_size, strides, threshold):
    fmeasures = []
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
                pred = (logits > threshold).float() # threshold the output activation map
                pred = logits.squeeze() # remove all dimensions of size 1 (batch dimension and channel dimension)

                prediction_unpadded = pred[0:window_size[0]-vertical_padding, 0:window_size[1]-horizontal_padding]
                img_prediction[coords[0]:coords[1],coords[2]:coords[3]] = prediction_unpadded
                
            img_prediction *= 255
            img_prediction = 255 - img_prediction

            fmeasure, pfmeasure, psnr, nrm, mpm, drd = compute_metrics(img_prediction, img_gt)
            fmeasures.append(fmeasure)
        
        print("final fmeasures", np.mean(fmeasures))
        

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

            # if i_f > shape[1]:
            #     i_f = shape[1] - 1
            #     i_0 = i_f - window_size[0]

            # if j_f > shape[0]:
            #     j_f = shape[0] - 1
            #     j_0 = j_f - window_size[1]
            
            yield (i_0,i_f, j_0,j_f), img[i_0:i_f, j_0:j_f]


def sliding_window_ignore_borders(img, strides, window_size):
    shape = img.shape
    tiles_per_row = int( np.ceil( shape[0] / float(strides[0]) ) )
    tiles_per_col = int( np.ceil( shape[1] / float(strides[1]) ) )

    maximum_row = strides[0] * int(np.floor( shape[0] / float(strides[0])))
    shift_row = (img.shape[0] - maximum_row) / 2

    maximum_col = strides[1] * int(np.floor( shape[1] / float(strides[1])))
    shift_col = (img.shape[1] - maximum_col) / 2

    for i in range(tiles_per_row):
        for j in range(tiles_per_col):
            i_0 = strides[0]*i + shift_row
            i_f = i_0 + window_size[0]

            j_0 = strides[1]*j + shift_col
            j_f = j_0 + window_size[1]

            # if i_f > shape[1]:
            #     i_f = shape[1] - 1
            #     i_0 = i_f - window_size[0]

            # if j_f > shape[0]:
            #     j_f = shape[0] - 1
            #     j_0 = j_f - window_size[1]
            
            yield (i_0,i_f, j_0,j_f), img[i_0:i_f, j_0:j_f]


if __name__ == '__main__':
    # TODO: add args functionality
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--network",type=str,default="fusionnet",help="choose between fusionnet & unet")
    # parser.add_argument("--batch_size",type=int,default=1,help="batch size")
    # parser.add_argument("--num_gpu",type=int,default=1,help="number of gpus")
    # args = parser.parse_args()

    main()
