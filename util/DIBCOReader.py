from torch.utils.data.dataset import Dataset
from torchvision import transforms
from glob import glob
import cv2
import numpy as np
from skimage.util.shape import view_as_blocks
import random
from PIL import Image

class DIBCODataset(Dataset):

    DIBCO = {
        2009: ['handwritten', 'printed'],
        2010: ['handwritten'],
        2011: ['handwritten', 'printed'],
        2012: ['handwritten'],
        2013: ['handwritten', 'printed'],
        2014: ['handwritten'],
        2016: ['handwritten']
    }

    def __init__(self, basepath="/home/dayvidwelles/phd/code/computer-vision-project/data/Dibco", years=[2009,2010,2011,2012,2013,2014], transform=None, windows_size=(256,256)):
        data_files = []
        for year in years:
            for subset in self.DIBCO[year]:
                dibco_imgs_path = "{}/{}/{}_GR/".format(basepath, year, subset)
                data_files.extend(glob(dibco_imgs_path + "*.png"))
        
        self.data_files = sorted(data_files)

        X_train = []
        Y_train = []

        for filename_gr in self.data_files:
            filename_gt = filename_gr.replace("GR", "GT")
                           
            img_gr = cv2.imread(filename_gr, cv2.IMREAD_GRAYSCALE)
            img_gt = cv2.imread(filename_gt, cv2.IMREAD_GRAYSCALE)

            # sliding window approach
            img_gr_h, img_gr_w = img_gr.shape
            new_gr_h, new_gr_w = int( windows_size[0] * round( float(img_gr_h) / windows_size[0] )), int( windows_size[1] * round( float(img_gr_w) / windows_size[1] ))

            img_gr = cv2.resize(img_gr, (new_gr_w, new_gr_h), interpolation = cv2.INTER_CUBIC)
            img_gt = cv2.resize(img_gt, (new_gr_w, new_gr_h), interpolation = cv2.INTER_CUBIC)

            # img_gr_patches = view_as_blocks(img_gr, windows_size).reshape(-1, windows_size[0], windows_size[1])
            # img_gt_patches = view_as_blocks(img_gt, windows_size).reshape(-1, windows_size[0], windows_size[1])
            img_gr_patches = self.blockshaped(img_gr, windows_size[0], windows_size[1])
            img_gt_patches = self.blockshaped(img_gt, windows_size[0], windows_size[1])

            X_train.extend(img_gr_patches)
            Y_train.extend(img_gt_patches)
            
        # convert to arrays
        X_train = np.asarray(X_train).astype('float32')
        Y_train = np.asarray(Y_train).astype('float32')

        # invert pixel values
        X_train = 255. - X_train
        X_train /= 255.

        # normalize target image and invert pixel values
        Y_train /= 255.
        Y_train = 1 - Y_train

        self.X_train = X_train
        self.Y_train = Y_train
        self.transform = transform
        self.target_transform = transform

    def blockshaped(self, arr, nrows, ncols):
        """
        Return an array of shape (n, nrows, ncols) where
        n * nrows * ncols = arr.size

        If arr is a 2D array, the returned array looks like n subblocks with
        each subblock preserving the "physical" layout of arr.
        """
        h, w = arr.shape
        return (arr.reshape(h//nrows, nrows, -1, ncols)
                .swapaxes(1,2)
                .reshape(-1, nrows, ncols))


    def unblockshaped(self, arr, h, w):
        """
        Return an array of shape (h, w) where
        h * w = arr.size

        If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
        then the returned array preserves the "physical" layout of the sublocks.
        """
        n, nrows, ncols = arr.shape
        return (arr.reshape(h//nrows, -1, nrows, ncols)
                .swapaxes(1,2)
                .reshape(h, w))
    
    def __getitem__(self, index):
        img_gr = Image.fromarray(self.X_train[index])
        img_gt = Image.fromarray(self.Y_train[index])
        # img_gr = self.X_train[index]
        # img_gt = self.Y_train[index]
        
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        if self.transform is not None:
            img_gr = self.transform(img_gr)
        
        random.seed(seed) # apply this seed to target tranfsorms
        if self.target_transform is not None:
            img_gt = self.target_transform(img_gt)

        return (img_gr, img_gt)

    # def __getitem__(self, index):
    #     filename_gr = self.data_files[index]
    #     filename_gt = self.data_files[index].replace("GR", "GT")

    #     img_gr = cv2.imread(filename_gr, cv2.IMREAD_GRAYSCALE)
    #     img_gt = cv2.imread(filename_gt, cv2.IMREAD_GRAYSCALE)

    #     return (img_gr, img_gt)

    def __len__(self):
        return len(self.X_train) # of how many examples(images?) you have


if __name__ == '__main__':
    # Define transforms (1)
    transformations = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    # Call the dataset
    # custom_dataset = MyCustomDataset(..., transformations)
    data_loader = DIBCODataset(transform=transformations)

