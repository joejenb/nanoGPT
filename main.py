import torch
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional.classification import multilabel_auroc, multilabel_accuracy

import argparse

import numpy as np
import os

import wandb

from nanoGPT import nanoGPTClassifier
from utils import get_lr

from configs.go_emotions_config import config

from utils import load_from_checkpoint, MakeConfig
from utils.data import _GO_EMOTIONS_LABELS as target_labels
from utils.data import get_data_loaders, log_bar

wandb.init(project="nanoGPTClassifier-GoEmotions", config=config)
config = MakeConfig(config)

def train(model, train_loader, epoch, optimiser):

    model.train()
    train_error = 0

    for iter_num, (ids, targets) in enumerate(train_loader):

        ids = ids.to(model.device)
        targets = targets.to(model.device)

        lr = get_lr((epoch * len(train_loader.dataset) // ids.shape[0]) + iter_num, optimiser.param_groups[0]['lr'], config)
        for param_group in optimiser.param_groups:
            param_group['lr'] = lr

        optimiser.zero_grad()

        _, loss = model(ids, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimiser.step()
        
        train_error += loss.item()

    wandb.log({
        "Train Error" : train_error / len(train_loader.dataset),
        "epoch" : epoch
    })


@torch.no_grad()
def test(model, test_loader, epoch):

    model.eval() 
    test_error = 0

    all_outputs = []
    all_targets = []

    for iter_num, (ids, targets) in enumerate(test_loader):
        ids = ids.to(model.device)
        targets = targets.to(model.device)

        logits, loss = model(ids, targets)
        
        test_error += loss.item()

        all_outputs.append(torch.sigmoid(logits))
        all_targets.append(targets)

    
    targets, outputs = torch.cat(all_targets), torch.cat(all_outputs)
    auroc = multilabel_auroc(outputs, targets, num_labels=len(target_labels), average=None, thresholds=None)
    accuracy = multilabel_accuracy(outputs, targets, num_labels=len(target_labels), average=None)

    log_bar(wandb, "Example Probabilities", target_labels, outputs[0], ["Class", "Probability"], epoch)
    log_bar(wandb, "Example Targets", target_labels, targets[0], ["Class", "Probability"], epoch)
    log_bar(wandb, "AUROC", target_labels, auroc, ["Class", "AUROC"], epoch)
    log_bar(wandb, "Accuracy", target_labels, accuracy, ["Class", "Accuracy"], epoch)
    
    wandb.log({
        "Test Error" : test_error / len(test_loader.dataset),
        "epoch" : epoch
    })

    return test_error / len(test_loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)

    args = parser.parse_args()
    PATH = args.data 

    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, val_loader, test_loader, num_classes = get_data_loaders(config, PATH)

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    checkpoint_location = f'checkpoints/{config.data_set}.ckpt'
    output_location = f'outputs/{config.data_set}.ckpt'

    model = nanoGPTClassifier(config, device).to(device)
    model = load_from_checkpoint(model, checkpoint_location)
    if config.block_size < model.config.block_size:
        model.crop_block_size(config.block_size)

    optimiser = model.configure_optimizers(config.weight_decay, config.learning_rate, config.betas)

    best_val_loss = float("inf")

    wandb.watch(model, log="all")

    for epoch in range(config.epochs):

        train(model, train_loader, epoch, optimiser)

        if not epoch % 5:
            loss = test(model, val_loader, epoch)

            if loss < best_val_loss:
                torch.save(model.state_dict(), output_location)

if __name__ == '__main__':
    main()