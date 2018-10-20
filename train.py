import numpy as np
import os
from util.DIBCOReader import DIBCODataset
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from models.AutoEncoder import AutoEncoder
import torch.optim as optim
from torch.autograd import Variable
from models import losses
from EarlyStopping import EarlyStopping
from metrics import f2_score, mse_score
from util.utils import save_checkpoint, load_checkpoint
import os

def criterion(logits, labels):
    return losses.F1ScoreLoss().forward(logits, labels)

def main():
    # Hyperparameters
    batch_size = 5
    epochs = 200
    threshold = 0.5
    early_stopping_patience = 20
    window_size = 256,256

    # Training informartion
    start_epoch = 0
    model_weiths_path = "checkpoints/"
    resume_training = True
    resume_checkpoint = model_weiths_path + "auto_encoder-epoch-51.pth"

    use_cuda = torch.cuda.is_available()

    # define the model and optimizer
    net = AutoEncoder(nb_layers=3).cuda()
    print(net)
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))

    # define the dataset and data augmentation operations

    training_set = DIBCODataset(years=[2009,2010,2011,2012,2013,2014],
    transform = transforms.Compose([
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor()]), window_size=window_size
    )

    testing_set = DIBCODataset(years=[2016], 
    transform = transforms.Compose([ 
                transforms.ToTensor()]), window_size=window_size
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

    # TODO: add training and testing log 
    print("Training on {} samples and testing on {} samples "
          .format(len(train_loader.dataset), len(test_loader.dataset)))

    early_stopper = EarlyStopping(mode='min', patience=early_stopping_patience)


    # optionally resume from a checkpoint
    if resume_training:
        if os.path.isfile(resume_checkpoint):
            print("=> loading checkpoint '{}'".format(resume_checkpoint))
            checkpoint = torch.load(resume_checkpoint)
            start_epoch = checkpoint['epoch']
            early_stopper.best = checkpoint['best_training_loss']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_checkpoint, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_checkpoint))

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

if __name__ == '__main__':
    # TODO: add args functionality
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--network",type=str,default="fusionnet",help="choose between fusionnet & unet")
    # parser.add_argument("--batch_size",type=int,default=1,help="batch size")
    # parser.add_argument("--num_gpu",type=int,default=1,help="number of gpus")
    # args = parser.parse_args()

    main()