from torch.utils.data.dataset import Dataset
from torchvision import transforms
from glob import glob


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

    def __init__(self, basepath="/home/dayvidwelles/phd/code/computer-vision-project/data/Dibco", years=[2009,2010,2011,2012,2013,2014], transforms=None):
        
        # X_train_paths = []
        # Y_train_paths = []
        # for year in years:
        #     for subset in self.DIBCO[year]:
        #         dibco_imgs_path = "{}/{}/{}_GR/".format(basepath, year, subset)
        #         dibco_ground_truth = "{}/{}/{}_GT/".format(basepath, year, subset)
        #         X_train_paths.extend(glob(dibco_imgs_path + "*.png"))
        #         Y_train_paths.extend(glob(dibco_ground_truth + "*.png"))

        data_files = []
        for year in years:
            for subset in self.DIBCO[year]:
                dibco_imgs_path = "{}/{}/{}_GR/".format(basepath, year, subset)
                data_files.extend(glob(dibco_imgs_path + "*.png"))
                
        self.data_files = sorted(self.data_files)

        
    def __getitem__(self, index):
        # stuff
        return (img, label)

    def __len__(self):
        return count # of how many examples(images?) you have


if __name__ == '__main__':
    # Define transforms (1)
    transformations = transforms.Compose([transforms.CenterCrop(100), transforms.ToTensor()])
    # Call the dataset
    # custom_dataset = MyCustomDataset(..., transformations)
    data_loader = DIBCODataset()