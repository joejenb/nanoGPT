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

To specify what size of nanoGPT to use for the backbone requires changing the config file that is imported into main.

## Results
### GoEmotions
#### Error
#### Accuracy
#### AUROC