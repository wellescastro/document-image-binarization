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

def criterion(logits, labels):
    return losses.F1ScoreLoss().forward(logits, labels)

def main():
    # Hyperparameters
    batch_size = 5
    epochs = 200
    threshold = 0.5
    train_log_step = 200

    use_cuda = torch.cuda.is_available()

    # define the model and optimizer
    net = AutoEncoder(nb_layers=3).cuda()
    print(net)
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))

    # define the dataset and data augmentation operations
    transformations = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    dataset = DIBCODataset(transform=transformations)

    train_loader = DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True # CUDA only
                         )


    # TODO: add training and validation log 
    # print("Training on {} samples and validating on {} samples "
    #       .format(len(train_loader.dataset), len(valid_loader.dataset)))

    for epoch in range(epochs):
        running_loss = 0
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

                running_loss += loss.item()
                if ind % train_log_step == train_log_step-1:    # print every 200 mini-batches
                    print('[%d, %5d] training loss: %.4f' %
                        (epoch + 1, ind + 1, running_loss / train_log_step))
                    running_loss = 0.0

if __name__ == '__main__':
    # TODO: add args functionality
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--network",type=str,default="fusionnet",help="choose between fusionnet & unet")
    # parser.add_argument("--batch_size",type=int,default=1,help="batch size")
    # parser.add_argument("--num_gpu",type=int,default=1,help="number of gpus")
    # args = parser.parse_args()

    main()