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

def criterion(logits, labels):
    return losses.F1ScoreLoss().forward(logits, labels)

def main():
    # Hyperparameters
    batch_size = 9
    epochs = 200
    threshold = 0.5
    early_stopping_patience = 20
    window_size = 256,256

    # Training informartion
    start_epoch = 0
    model_weiths_path = "checkpoints/"
    resume_training = False
    resume_checkpoint = model_weiths_path + "auto_encoder-epoch-51.pth"

    use_cuda = torch.cuda.is_available()

    # define the model and optimizer
    net = AutoEncoder(nb_layers=3).cuda()
    print(net)
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))

    # define the dataset and data augmentation operations

    training_transforms = transforms.Compose([
                transforms.ToPILImage(mode='L'),
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor()])

    training_set = DIBCODataset(years=[2009,2010,2011,2012,2013,2014],
    transform = training_transforms, window_size=window_size
    )

    testing_transforms = transforms.Compose([ 
                transforms.ToPILImage(mode='L'),
                transforms.ToTensor()])
    testing_set = DIBCODataset(years=[2016], 
    transform = testing_transforms, window_size=window_size
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
        start_epoch, early_stopper.best = load_checkpoint(net, optimizer, resume_checkpoint)

    for epoch in range(start_epoch, epochs):
        training_metrics = {'loss':0, 'mse':0, 'f1score':0}
        testing_metrics = {'loss':0, 'mse':0, 'f1score':0}

        # perform training
        net.train()
        for ind, (inputs, target) in enumerate(train_loader):
            if use_cuda:
                inputs = inputs.cuda()
                target = target.cuda()

            inputs, target = Variable(inputs), Variable(target)

            # forward
            logits = net.forward(inputs)

            # backward + optimize
            loss = criterion(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_metrics['loss'] += loss.item()
            training_metrics['mse'] += mse_score(logits, target)

            # get thresholded prediction and compute the f1-score per patches
            training_metrics['f1score'] += f2_score(target.view(-1, window_size[0] * window_size[1]), logits.view(-1, window_size[0] * window_size[1]), threshold=threshold).item()
        
        # get the average training loss
        training_metrics['loss'] /= len(train_loader)
        training_metrics['mse'] /= len(train_loader)
        training_metrics['f1score'] /= len(train_loader)
        
        # perform validation 
        net.eval()
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
                testing_metrics['mse'] += mse_score(logits, target)

                # get thresholded prediction and compute the f1-score per patches
                testing_metrics['f1score'] += f2_score(target.view(-1, window_size[0] * window_size[1]), logits.view(-1, window_size[0] * window_size[1]), threshold=threshold).item()
        
        # get the average of the metrics
        testing_metrics['loss'] /= len(test_loader)
        testing_metrics['mse'] /= len(test_loader)
        testing_metrics['f1score'] /= len(test_loader)

        print('[%d, %d] train_loss: %.4f test_loss: %.4f train_mse: %.4f test_mse: %.4f train_f1score: %.4f test_f1score: %.4f current patience: %d' %
                    (epoch + 1, epochs, training_metrics['loss'], testing_metrics['loss'], training_metrics['mse'], testing_metrics['mse'], training_metrics['f1score'], testing_metrics['f1score'], early_stopper.patience))

        # save checkpoint
        is_best = training_metrics['loss'] < early_stopper.best

        if early_stopper.step(training_metrics['loss']) == True:
            print("Early stopping triggered!")
            break

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'best_training_loss': early_stopper.best,
            'optimizer' : optimizer.state_dict(),
        }, is_best, model_weiths_path)
    

    load_weights(net, "checkpoints/model_best.pth")

    # list_of_patches = testing_set.list_of_patches
    # old_sum = sum(list_of_patches[:0]) # empty
    # for i in range(len(testing_set.list_of_patches)):
    #     # get the patches of each testing image
    #     input_patches = testing_set.X_train[old_sum:sum(list_of_patches[:i+1])]
    #     target_patches = testing_set.Y_train[old_sum:sum(list_of_patches[:i+1])]

    #     # evaluation here
    #     inputs, target = torch.from_numpy(input_patches).float().to('cuda'), torch.from_numpy(target_patches).float().to('cuda')
    #     inputs, target = inputs.unsqueeze(1), target.unsqueeze(1)

    #     # forward
    #     logits = net.forward(inputs)
    #     pred = (logits > threshold).float()

    #     print(len(input_patches))
    #     print(list_of_patches[i])
    #     print(pred.shape)

    #     pred = pred.view(list_of_patches[i], window_size[0], window_size[1])
    #     print(pred.cpu().numpy().shape)

    #     testing_set.unblockshaped(input_patches, window_size[0], window_size[1])
        
    #     old_sum = sum(testing_set.list_of_patches[:i+1])

    #     exit()


    strides = (128, 128)

    # starting final evaluation using the reconstructed image
    with torch.no_grad():
        for filename_gr in testing_set.data_files:
                filename_gt = filename_gr.replace("GR", "GT")
                                
                img_gr = cv2.imread(filename_gr, cv2.IMREAD_GRAYSCALE)
                img_gt = cv2.imread(filename_gt, cv2.IMREAD_GRAYSCALE)

                #preprocessing
                img_gr = 255 - img_gr

                dim_row, dim_col = img_gr.shape

                # tiles_per_row, tiles_per_col = int(dim_row // float(strides[0])), int(dim_row // float(strides[1]))

                img_prediction = np.zeros(img_gr.shape, dtype=np.uint8)
                for coords, patch in sliding_window(img_gr, strides, window_size):
                    patch_size = patch.shape

                    vertical_padding = (window_size[0] - patch_size[0])
                    horizontal_padding = (window_size[1] - patch_size[1])
                    patch_gr = np.pad(patch, ((0,vertical_padding),(0,horizontal_padding)),mode='constant')
                    patch_gr_with_channel = np.expand_dims(patch_gr, 2)
        
                    inputs = testing_transforms(patch_gr_with_channel).cuda()
                    inputs = inputs.unsqueeze(1)

                    # forward
                    logits = net(inputs)
                    pred = (logits > threshold).float() # threshold the output activation map
                    pred = logits.squeeze() # remove all dimensions of size 1 (batch dimension and channel dimension)

                    prediction_unpadded = pred[0:window_size[0]-vertical_padding, 0:window_size[1]-horizontal_padding]
                    img_prediction[coords[0]:coords[1],coords[2]:coords[3]] = prediction_unpadded
                    

                Image.fromarray(img_prediction, mode='1').show()
                exit()

                inputs = torch.from_numpy(patch_gr).float().to('cuda')

    


def sliding_window(img, strides, window_size):
    shape = img.shape
    tiles_per_row = int( np.ceil( shape[0] / float(strides[0]) ) )
    tiles_per_col = int( np.ceil( shape[1] / float(strides[1]) ) )

    for i in xrange(tiles_per_row):
        for j in xrange(tiles_per_col):
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

    for i in xrange(tiles_per_row):
        for j in xrange(tiles_per_col):
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