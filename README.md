computer-vision-project

# Requerimentos
Python (versão utilizada 2.7)
Pytorch (versão utilizada 0.4.1)
CUDA e cuDNN

Testado no Linux Mint e Ubuntu 

# Treinamento

python train.py


# Avaliação

python eval.py --model weights/DIBCO2016-WHOLE_AUG/best_model.pth --year 2016

python eval.py --model weights/DIBCO2016-WHOLE_AUG/best_model.pth --year 2016 --save_dir "dibco2016_imagens_geradas"



# Binarização

python binarize.py --model weights/DIBCO2014-WHOLE_AUG/best_model.pth --image_path data/Dibco/2016/handwritten_GR/10.png
