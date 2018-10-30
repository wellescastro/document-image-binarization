import os
from glob import glob
from imgaug import augmenters as iaa
import cv2
from PIL import Image

def main():
    DIBCO = {
        2009: ['handwritten', 'printed'],
        2010: ['handwritten'],
        2011: ['handwritten', 'printed'],
        2012: ['handwritten'],
        2013: ['handwritten', 'printed'],
        2014: ['handwritten'],
        2016: ['handwritten']
    }

    basepath = "Dibco"

    for year, subsets in DIBCO.items():
        for subset in subsets:
            dibco_imgs_path = "{}/{}/{}_GR/".format(basepath, year, subset)

            input_images_paths = glob(dibco_imgs_path + "*.png")
            input_images = list(map(lambda i:cv2.imread(i, 0), input_images_paths))
            target_images = list(map(lambda i:cv2.imread(i.replace("GR", "GT"), 0), input_images_paths))

            # start augmentation for each folder 
            dst_dir_gr = dibco_imgs_path.replace("_GR", "_GR_aug")
            dst_dir_gt = dibco_imgs_path.replace("_GR", "_GT_aug")
            create_dir(dst_dir_gr)
            create_dir(dst_dir_gt)

            seq = iaa.Sequential([
                iaa.Fliplr(0.5), 
                iaa.Affine(scale={"y": (0.5, 1.5)})
                ])

            number_of_samples = 3

            for i in range(number_of_samples):
                # Convert the stochastic sequence of augmenters to a deterministic one.
                # The deterministic sequence will always apply the exactly same effects to the images.
                seq_det = seq.to_deterministic() # call this for each batch again, NOT only once at the start
                images_aug = seq_det.augment_images(input_images)
                heatmaps_aug = seq_det.augment_images(target_images)
            
                for gr_img, gt_img, gr_img_name in zip(images_aug, heatmaps_aug, input_images_paths):
                    img_name = gr_img_name.split("/")[-1].replace(".", "_{}.".format(i))
                    gr_aug_img_path = dst_dir_gr + img_name
                    gt_aug_img_path = dst_dir_gt + img_name.replace("GR", "GT")
                    cv2.imwrite(gr_aug_img_path, gr_img)
                    cv2.imwrite(gt_aug_img_path, gt_img)

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == '__main__':
    main()