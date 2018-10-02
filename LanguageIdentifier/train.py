import torch
import os
import sys
import numpy as np
import math
import datetime

from torchtext.data import Iterator
from torch.optim import adam

from LanguageIdentifier.model import Model, RecurrentModel, CharModel, CNNRNN
from LanguageIdentifier.test import test
from LanguageIdentifier.utils import save_model, PAD_TOKEN


def train(par_optimizer: adam=None, model: Model=None,
          training_data: Iterator=None, validation_data: Iterator=None, testing_data: Iterator=None,
          learning_rate: float=1e-3, epochs: int=0, resume_state: dict=None, resume: str="",
          log_frequency: int=0, eval_frequency: int=0, model_type="", output_dir="", scheduler=None,
          level: str='char',
          **kwargs):

    # get command line arguments
    cfg = locals().copy()

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if not resume:
        best_train_acc, best_val_acc, best_test_acc = 0, 0, 0
        start_epoch = 0
    else:
        start_epoch = resume_state['epoch']
        training_acc = resume_state['train_acc']
        validation_acc = resume_state['val_acc']
        test_acc = resume_state['test_acc']
        best_val_acc = validation_acc

    print(datetime.datetime.now(), " Training starts.")
    batch_accuracies, epoch_accuracies = [], []
    for i in range(start_epoch, epochs):

        if scheduler:
            scheduler.step()

        
        epoch_losses = []
        for j, batch in enumerate(iter(training_data)):
            model.train()
            par_optimizer.zero_grad()

            # We take the characters as input to the network, and the languages
            # as target
            if level == 'char':
                sequence = batch.characters[0]
                lengths = batch.characters[1]
                target = batch.language
                # NLLLoss for using the log_softmax in the recurrent model
                pad_idx = training_data.dataset.fields['characters'].vocab.stoi[PAD_TOKEN]
                loss_function = torch.nn.NLLLoss(ignore_index=pad_idx)
            else:
                sequence = batch.paragraph[0]
                lengths = batch.paragraph[1]
                char_lengths = batch.paragraph[2]
                target = batch.language
                # NLLLoss for using the log_softmax in the recurrent model
                pad_idx = training_data.dataset.fields['paragraph'].vocab.stoi[PAD_TOKEN]
                loss_function = torch.nn.NLLLoss(ignore_index=pad_idx)

            # characters = batch.characters[0]
            # languages = batch.language
            batch_size = sequence.shape[0]

            if isinstance(model, RecurrentModel):
                predictions = model.forward(sequence, batch.characters[1])
            elif isinstance(model, CharModel):
                predictions = model.forward(sequence)
            else:
                predictions = model.forward(sequence, char_lengths, lengths)
            loss = loss_function(predictions, target.squeeze(1))
            epoch_losses.append(loss.item())

            _, predicted_languages = torch.topk(predictions, 1)

            # Save data needed to calculate accuracy for later
            batch_accuracy = target.eq(predicted_languages).sum().item() / batch_size
            batch_accuracies.append(batch_accuracy)
            epoch_accuracies.append(batch_accuracy)

            # Update the weights
            loss.backward()
            par_optimizer.step()

            if (j + 1) % cfg["log_frequency"] == 0:
                if scheduler:
                    lr = scheduler.get_lr()[0]
                else:
                    lr = cfg["learning_rate"]
                print(datetime.datetime.now(), "Logging: Epoch: {} | Iter: {} | Loss: {} | Batch accuracy: {} "
                                               "| LR: {}".format(
                    i, j, round(loss.item(), 4), round(batch_accuracies[-1], 3), round(lr, 5))
                      )

            if (j + 1) % cfg["eval_frequency"] == 0:
                train_accuracy = np.array(batch_accuracies).mean()
                batch_accuracies = []
                validation_accuracy = test(model, validation_data, level=level)
                print(datetime.datetime.now(), " Evaluation: Epoch: {} | Iter: {} | Loss: {} | "
                      "Av. Batch Train accuracy: {}| Validation accuracy {} ".format(
                    i, j, round(loss.item(), 4), round(train_accuracy, 2), round(validation_accuracy, 2)))
                if validation_accuracy > best_val_acc:
                    best_train_acc = train_accuracy
                    test_accuracy = test(model, testing_data, True, level=level)
                    best_test_acc = test_accuracy
                    best_val_acc = validation_accuracy
                    output_dir = cfg["output_dir"]
                    save_model(output_dir,
                               {
                                   'epoch': i,
                                   'state_dict': model.state_dict(),
                                   'train_acc': train_accuracy,
                                   'val_acc': validation_accuracy,
                                   'test_acc': test_accuracy,
                                   'optimizer': par_optimizer.state_dict(),
                               },
                               filename=cfg["model_type"] + "_best_model.pth.tar")

        train_accuracy = np.array(epoch_accuracies).mean()
        epoch_accuracies = []
        validation_accuracy = test(model, validation_data, level=level)

        print(datetime.datetime.now(), " Epoch: {} finished | Average loss: {} | "
              "Av. Batch Train accuracy: {} | Validation accuracy: {}".format(
              i + 1, round(np.mean(np.array(epoch_losses)), 2), round(train_accuracy, 2), round(validation_accuracy, 2)
        ))

    print(datetime.datetime.now(), " Done training.")
    print("Best model: Train accuracy: {} | Validation accuracy: {} | Test accuracy: {}".format(
          round(best_train_acc, 2), round(best_val_acc, 2), round(best_test_acc, 2)))
