from torch.utils.data.dataset import Dataset
from torchvision import transforms
from glob import glob
import cv2
import numpy as np
from skimage.util.shape import view_as_blocks

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

    def __init__(self, basepath="/home/dayvidwelles/phd/code/computer-vision-project/data/Dibco", years=[2009,2010,2011,2012,2013,2014], transforms=None, windows_size=(256,256)):
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

            img_gr_patches = view_as_blocks(img_gr, windows_size).reshape(-1, windows_size[0], windows_size[1])
            img_gt_patches = view_as_blocks(img_gt, windows_size).reshape(-1, windows_size[0], windows_size[1])

            X_train.extend(img_gr_patches)
            Y_train.extend(img_gt_patches)

        # convert to arrays
        X_train = np.asarray(X_train).astype('float32')
        Y_train = np.asarray(Y_train).astype('float32')

        # invert pixel values
        X_train = 1 - X_train

        # normalize target image and invert pixel values
        Y_train /= 255.
        Y_train = 1 - Y_train

        self.X_train = X_train
        self.Y_train = Y_train

    def patchify(self, img, patch_shape):
        img = np.ascontiguousarray(img)  # won't make a copy if not needed
        X, Y = img.shape
        x, y = patch_shape
        shape = ((X-x+1), (Y-y+1), x, y) # number of patches, patch_shape
        # The right strides can be thought by:
        # 1) Thinking of `img` as a chunk of memory in C order
        # 2) Asking how many items through that chunk of memory are needed when indices
        #    i,j,k,l are incremented by one
        strides = img.itemsize*np.array([Y, 1, Y, 1])
        return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)

    
    def __getitem__(self, index):
        filename_gr = self.data_files[index]
        filename_gt = self.data_files[index].replace("GR", "GT")

        img_gr = cv2.imread(filename_gr, cv2.IMREAD_GRAYSCALE)
        img_gt = cv2.imread(filename_gt, cv2.IMREAD_GRAYSCALE)

        return (img_gr, img_gt)    

    # def __getitem__(self, index):
    #     filename_gr = self.data_files[index]
    #     filename_gt = self.data_files[index].replace("GR", "GT")

    #     img_gr = cv2.imread(filename_gr, cv2.IMREAD_GRAYSCALE)
    #     img_gt = cv2.imread(filename_gt, cv2.IMREAD_GRAYSCALE)

    #     return (img_gr, img_gt)

    def __len__(self):
        return count # of how many examples(images?) you have


if __name__ == '__main__':
    # Define transforms (1)
    transformations = transforms.Compose([transforms.CenterCrop(100), transforms.ToTensor()])
    # Call the dataset
    # custom_dataset = MyCustomDataset(..., transformations)
    data_loader = DIBCODataset()
    data_loader.__getitem__(0)