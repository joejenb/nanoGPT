import torch
import torchvision
import pandas as pd

from torchvision import transforms
from torch.utils.data import random_split

from datasets import load_dataset
from transformers import GPT2Tokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader

_GO_EMOTIONS_LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

class GoEmotionDataset(Dataset):
    def __init__(self, texts, targets):
        self.texts = texts
        self.targets = targets
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = self.texts[index]
        targets = self.targets[index]
        
        return {
            "text": text, 
            "targets": targets
        }

class ClassificationCollator:
    def __init__(self, tokenizer, targets_encoder, max_sequence_len=None):

        self.tokenizer = tokenizer
        self.max_sequence_len =  max_sequence_len if max_sequence_len else tokenizer.model_max_length
        self.targets_encoder = targets_encoder

    def __call__(self, data):
        texts = [inst['text'] for inst in data]
        targets = [inst['targets'] for inst in data]

        inputs = self.tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)
        #inputs.update({'targets': torch.tensor(targets)})

        return inputs['input_ids'], torch.tensor(targets, dtype=torch.long)

def one_hot_targets(data, num_labels):
    dict_targets = []
    for i in range(len(data)):
        d = dict(zip(range(num_labels), [0]*num_labels))
        labels = data.loc[i]["labels"]
        for label in labels:
            d[label] = 1
        dict_targets.append(d)

    data_targets = pd.DataFrame(dict_targets)
    return data_targets

def get_tokenizer(tokenizer_name):
    if tokenizer_name == "GPT2Tokenizer":
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path="gpt2")
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_data_loaders(config, PATH):
    mlb = MultiLabelBinarizer()
    if config.data_set == "GOEMOTIONS":
        num_classes = len(_GO_EMOTIONS_LABELS)
        tokenizer = get_tokenizer(config.tokenizer)
        targets_encoder = list(range(num_classes))

        collator = ClassificationCollator(tokenizer, targets_encoder, config.max_sequence_len)
        go_emotions = load_dataset("go_emotions")
        data = go_emotions.data

        train_set = go_emotions.data["train"].to_pandas()
        val_set = go_emotions.data["validation"].to_pandas()
        test_set = go_emotions.data["test"].to_pandas()

        train_set = GoEmotionDataset(train_set.text.tolist(), pd.DataFrame(mlb.fit_transform(train_set['labels'])).values.tolist())
        val_set = GoEmotionDataset(val_set.text.tolist(), pd.DataFrame(mlb.fit_transform(val_set['labels'])).values.tolist())
        test_set = GoEmotionDataset(test_set.text.tolist(), pd.DataFrame(mlb.fit_transform(test_set['labels'])).values.tolist())

    train_loader = DataLoader(train_set, collate_fn=collator, batch_size=config.batch_size, shuffle=False)
    val_loader = DataLoader(val_set, collate_fn=collator, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, collate_fn=collator, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, num_classes
