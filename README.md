# Binarização de Documentos Manuscritos Históricos

Esse repositório contém scripts implementados em PyTorch para o treinamento de um modelo Autoencoder com camadas convolucionais e convoluções transpostas [proposto aqui](https://www.sciencedirect.com/science/article/abs/pii/S0031320318303091) usando a medida-F como função de perda. Esse projeto foi realizado como parte da avaliação da disciplina de Visão Computacional no Doutorado em Ciência da Computação do CIn-UFPE.

# Bases de Dados
* Bases de dados da DIBCO 2009 até DIBCO 2017 
(uma das bases de dados é separada para validação e o restante são usadas para o treinamento do modelo)

# Requerimentos
* Python (versão utilizada 2.7)
* Pytorch (versão utilizada 0.4.1)

Testado no Linux Mint e Ubuntu 

# Treinamento
O treinamento do modelo pode ser realizado com o seguinte comando:
``` 
python train.py
```

# Teste

A avaliação do modelo com uma base de dados de validação pode ser feita com o seguinte comando:
``` 
python eval.py --model weights/DIBCO2016-WHOLE_AUG/best_model.pth --year 2016
``` 
Você também pode especificar o diretório para visualizar as imagens binarizadas:
``` 
python eval.py --model weights/DIBCO2016-WHOLE_AUG/best_model.pth --year 2016 --save_dir "dibco2016_imagens_geradas"
``` 

Para binarizar uma única imagem, o seguinte comando pode ser usado:
``` 
python binarize.py --model weights/DIBCO2014-WHOLE_AUG/best_model.pth --image_path data/Dibco/2016/handwritten_GR/10.png
``` 
