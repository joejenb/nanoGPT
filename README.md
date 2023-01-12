# nanoGPTClassifier

Text classifier with an altered version of [nanoGPT](https://github.com/karpathy/nanoGPT) as its backbone for simplicity and efficiency. Facilitates both multi-class and multi-label classification. Currently trains on GoEmotions dataset for text sentiment classification.
## Install

Dependencies:

- [pytorch](https://pytorch.org)
- [numpy](https://numpy.org/install/)
- `pip install datasets`
- `pip install wandb`
- `pip install torchmetrics`

## Usage

```
$ python main.py
```

Please note, to specify what size of nanoGPT to use for the backbone requires changing parameters in the config file imported into main.

## Results

All results collected are using nanoGPT with 12 layers, 12 heads and 768 embeddings.
### GoEmotions
#### Example

Example targets and probabilities for a given validation instance after first epoch

![Example_Targets_Start](https://user-images.githubusercontent.com/30124151/212054607-8876a394-324b-4ce8-9afd-ed061009d7fa.png)
![Example_Probabilities_Start](https://user-images.githubusercontent.com/30124151/212054622-b30497f7-7321-4fd4-ac9e-638151f510ce.png)

Example targets and probabilities for a given validation instance after fifth epoch

![Example_Targets_End](https://user-images.githubusercontent.com/30124151/212054721-1028555d-1d96-4754-ad4b-914fcd44867f.png)
![Example_Probabilities_End](https://user-images.githubusercontent.com/30124151/212054749-51b6a4c9-85c4-4c81-8d92-4649230db460.png)

#### Error

![Train_Error](https://user-images.githubusercontent.com/30124151/212054917-74fc68b2-deeb-4aa6-90b8-6bc94dba1a5b.png)
![Validation_Error](https://user-images.githubusercontent.com/30124151/212054936-3be60d95-18c8-4ee5-9a58-d525f96f6ce4.png)

#### Accuracy

![Accuracy](https://user-images.githubusercontent.com/30124151/212054971-ec21a53d-7e9a-4781-92cd-dba988cd1e05.png)

#### AUROC

![AUROC](https://user-images.githubusercontent.com/30124151/212055009-1e673f2c-c7cf-42c4-86fd-d8d2de6e97a7.png)
