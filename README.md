computer-vision-project


# Training

python train.py


# Evaluation

python eval.py --model weights/DIBCO2016-WHOLE_AUG/best_model.pth --year 2016


# Binarization

python binarize.py --model weights/DIBCO2014-WHOLE_AUG/best_model.pth --image_path data/Dibco/2016/handwritten_GR/10.png