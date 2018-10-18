import numpy as np
import os
from util.DIBCOReader import DIBCODataset
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from models.AutoEncoder import AutoEncoder

def main():
    # Hyperparameters
    batch_size = 3
    epochs = 50
    threshold = 0.5

    use_cuda = torch.cuda.is_available()

    # net = unet_origin.UNetOriginal((3, *img_resize))
    # classifier = nn.classifier.CarvanaClassifier(net, epochs)
    # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.99)

    transformations = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    dataset = DIBCODataset(transform=transformations)

    train_loader = DataLoader(dataset,
                          batch_size=5,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True # CUDA only
                         )


    # TODO: add training and validation log 
    # print("Training on {} samples and validating on {} samples "
    #       .format(len(train_loader.dataset), len(valid_loader.dataset)))

    for epoch in range(epochs):
        for ind, (inputs, target) in enumerate(train_loader):
                if self.use_cuda:
                    inputs = inputs.cuda()
                    target = target.cuda()
                inputs, target = Variable(inputs), Variable(target)

                # # forward
                # logits = self.net.forward(inputs)
                # probs = F.sigmoid(logits)
                # pred = (probs > threshold).float()

                # # backward + optimize
                # loss = self._criterion(logits, target)
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()

        exit()


if __name__ == '__main__':
    # TODO: add args functionality
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--network",type=str,default="fusionnet",help="choose between fusionnet & unet")
    # parser.add_argument("--batch_size",type=int,default=1,help="batch size")
    # parser.add_argument("--num_gpu",type=int,default=1,help="number of gpus")
    # args = parser.parse_args()

    main()