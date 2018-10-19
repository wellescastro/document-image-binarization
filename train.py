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
import shutil
from EarlyStopping import EarlyStopping

def criterion(logits, labels):
    return losses.F1ScoreLoss().forward(logits, labels)

def save_checkpoint(state, is_best, checkpoint_dir):
    filename = os.path.join(checkpoint_dir, 'auto_encoder-epoch-{}.pth'.format(state["epoch"]))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, checkpoint_dir + 'model_best.pth.tar')

def main():
    # Hyperparameters
    batch_size = 5
    epochs = 200
    threshold = 0.5
    early_stopping_patience = 20

    # Training informartion
    train_log_step = 200
    save_freq = 1
    model_weiths_path = "checkpoints/"

    use_cuda = torch.cuda.is_available()

    # define the model and optimizer
    net = AutoEncoder(nb_layers=3).cuda()
    print(net)
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))

    # define the dataset and data augmentation operations

    training_set = DIBCODataset(years=[2009,2010,2011,2012,2013,2014],
    transform = transforms.Compose([
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor()])
    )

    testing_set = DIBCODataset(years=[2016], 
    transform = transforms.Compose([ 
                transforms.ToTensor()])
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

    best_training_loss = 99999
    early_stopper = EarlyStopping(mode='min', patience=20)

    for epoch in range(epochs):
        training_loss = 0
        testing_loss = 0
        
        # perform training
        net.train()
        for ind, (inputs, target) in enumerate(train_loader):
            if use_cuda:
                inputs = inputs.cuda()
                target = target.cuda()

            inputs, target = Variable(inputs), Variable(target)

            # forward
            logits = net.forward(inputs)
            pred = (logits > threshold).float()

            # backward + optimize
            loss = criterion(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
        
        # get the average training loss
        training_loss /= len(train_loader)
        
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

                testing_loss += loss

                # get thresholded prediction and compute the f1-score
                pred = (logits > threshold).float()
        
        # get the average validation loss
        testing_loss /= len(test_loader)

        print('[%d, %d] train loss: %.4f test loss: %.4f current stopping patience: %.4f' %
                    (epoch + 1, epochs, training_loss, testing_loss, early_stopper.patience))

        # remember best training loss and save checkpoint
        is_best = training_loss < best_training_loss

        if early_stopper.step(training_loss) == True:
            print("Training finished!")
            break

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'best_training_loss': early_stopper.best,
            'optimizer' : optimizer.state_dict(),
        }, is_best, model_weiths_path)

        # reset for the next epoch
        training_loss = 0
        testing_loss = 0

if __name__ == '__main__':
    # TODO: add args functionality
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--network",type=str,default="fusionnet",help="choose between fusionnet & unet")
    # parser.add_argument("--batch_size",type=int,default=1,help="batch size")
    # parser.add_argument("--num_gpu",type=int,default=1,help="number of gpus")
    # args = parser.parse_args()

    main()