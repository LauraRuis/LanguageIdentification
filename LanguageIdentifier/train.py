import torch
import os
import sys
import numpy as np

from torchtext.data import Iterator
from torch.optim import adam

from LanguageIdentifier.model import Model, RecurrentModel
from LanguageIdentifier.test import test
from LanguageIdentifier.utils import save_model


def train(optimizer: adam=None, model: Model=None,
          training_data: Iterator=None, validation_data: Iterator=None, testing_data: Iterator=None,
          learning_rate: float=1e-3, epochs: int=0, resume_state: dict=None, resume: str="",
          log_frequency: int=0, eval_frequency: int=0,
          **kwargs):

    # get command line arguments
    cfg = locals().copy()

    # NLLLoss for using the log_softmax in the recurrent model
    loss_function = torch.nn.NLLLoss()

    if not resume:
        best_train_acc, best_val_acc, best_test_acc = 0, 0, 0
        start_epoch = 0
    else:
        start_epoch = resume_state['epoch']
        training_acc = resume_state['train_acc']
        validation_acc = resume_state['val_acc']
        test_acc = resume_state['test_acc']

    print("Training starts.")
    for i in range(start_epoch, epochs):
        model.train()
        epoch_losses = []
        for j, batch in enumerate(iter(training_data)):
            optimizer.zero_grad()

            # We take the characters as input to the network, and the languages
            # as targets
            characters = torch.autograd.Variable(batch.characters[0])
            languages = batch.language
            if isinstance(model, RecurrentModel):
                predictions = model.forward(characters, batch.characters[1])
            else:
                predictions = model.forward(characters)
            loss = loss_function(predictions, languages.squeeze(1))
            epoch_losses.append(loss.item()) 

            # Update the weights
            loss.backward()
            optimizer.step()

            if j % cfg["log_frequency"] == 0:
                print("Logging: Epoch: {} | Iter: {} | Loss: {} ".format(i, j, loss.item()))

            if j % cfg["eval_frequency"] == 0:
                train_accuracy = test(model, training_data)
                validation_accuracy = test(model, validation_data)
                print("Evaluation: Epoch: {} | Iter: {} | Loss: {} | Train accuracy: {}| Validation accuracy {} ".format(
                    i, j, loss.item(), train_accuracy, validation_accuracy))

        train_accuracy = test(model, training_data)
        validation_accuracy = test(model, validation_data)

        print("Epoch: {} finished | Average loss: {} | Train accuracy: {} | Validation accuracy: {}".format(
              i + 1, np.mean(np.array(epoch_losses)), train_accuracy, validation_accuracy
        ))

        if validation_accuracy > best_val_acc:
            best_train_acc = train_accuracy
            test_accuracy = test(model, testing_data)
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
                           'optimizer': optimizer.state_dict(),
                       })

    print("Done training.")
    print("Best model: Train accuracy: {} | Validation accuracy: {} | Test accuracy: {}".format(
          best_train_acc, best_val_acc, best_test_acc))
