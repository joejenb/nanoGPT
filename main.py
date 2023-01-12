import torch
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional.classification import multilabel_auroc, multilabel_accuracy

import numpy as np
import wandb
import os

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
    
    return train_error

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
    auroc = multilabel_auroc(outputs, targets, num_labels=len(target_labels), average="micro", thresholds=None)
    accuracy = multilabel_accuracy(outputs, targets, num_labels=len(target_labels), average="micro")

    outputs = log_bar(wandb, "Example Probabilities", target_labels, outputs[0], ["Class", "Probability"], epoch)
    targets = log_bar(wandb, "Example Targets", target_labels, targets[0], ["Class", "Probability"], epoch)
    auroc = log_bar(wandb, "AUROC", target_labels, auroc, ["Class", "AUROC"], epoch)
    accuracy = log_bar(wandb, "Accuracy", target_labels, accuracy, ["Class", "Accuracy"], epoch)

    return test_error, outputs, targets, auroc, accuracy

def main():
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, val_loader, test_loader, num_classes = get_data_loaders(config)

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

        train_error = 0#train(model, train_loader, epoch, optimiser)

        if not epoch % 5:
            val_error, outputs, targets, auroc, accuracy = test(model, val_loader, epoch)

            #  wandb seems to overwrite tables and charts so name according to epoch
            wandb.log({
                "Train Error" : train_error / len(train_loader.dataset),
                "Validation Error" : val_error / len(val_loader.dataset),
                f'Example Probabilities {epoch}': outputs,
                f'Example Targets {epoch}': targets,
                f'AUROC {epoch}': auroc,
                f'Accuracy {epoch}': accuracy,
            })

            if val_error < best_val_loss:
                torch.save(model.state_dict(), output_location)
        else:
            wandb.log({"Train Error" : train_error / len(train_loader.dataset)})

if __name__ == '__main__':
    main()