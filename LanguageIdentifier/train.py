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
          log_frequency: int=0, eval_frequency: int=0, model_type="",
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
    batch_accuracies, epoch_accuracies = [], []
    for i in range(start_epoch, epochs):
        model.train()
        epoch_losses = []
        for j, batch in enumerate(iter(training_data)):

            optimizer.zero_grad()

            # We take the characters as input to the network, and the languages
            # as targets

            # use of Variable not necessary
            # characters = torch.autograd.Variable(batch.characters[0])
            characters = batch.characters[0]
            languages = batch.language
            if isinstance(model, RecurrentModel):
                predictions = model.forward(characters, batch.characters[1])
            else:
                predictions = model.forward(characters)
            loss = loss_function(predictions, languages.squeeze(1))
            epoch_losses.append(loss.item())

            _, predicted_languages = torch.topk(predictions, 1)

            # Save data needed to calculate accuracy for later
            batch_accuracy = languages.eq(predicted_languages)
            batch_accuracies.extend(batch_accuracy)
            epoch_accuracies.extend(batch_accuracy)

            # Update the weights
            loss.backward()
            optimizer.step()

            if (j + 1) % cfg["log_frequency"] == 0:
                print("Logging: Epoch: {} | Iter: {} | Loss: {3.3f} | Batch accuracy: {3.3f}".format(
                    i, j, loss.item(), batch_accuracies[-1].item()))

            if (j + 1) % cfg["eval_frequency"] == 0:
                train_accuracy = numpy.array([sample.item() for sample in batch_accuracies]).mean()
                batch_accuracies = []
                validation_accuracy = test(model, validation_data)
                print("Evaluation: Epoch: {} | Iter: {} | Loss: {3.3f} | "
                      "Av. Batch Train accuracy: {3.3f}| Validation accuracy {3.3f} ".format(
                    i, j, loss.item(), train_accuracy, validation_accuracy))

        train_accuracy = numpy.array([sample.item() for sample in epoch_accuracies]).mean()
        epoch_accuracies = []
        validation_accuracy = test(model, validation_data)

        print("Epoch: {} finished | Average loss: {3.3f} | "
              "Av. Batch Train accuracy: {3.3f} | Validation accuracy: {3.3f}".format(
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
                       },
                       filename=cfg["model_type"] + "_best_model.pth.tar")

    print("Done training.")
    print("Best model: Train accuracy: {} | Validation accuracy: {} | Test accuracy: {}".format(
          best_train_acc, best_val_acc, best_test_acc))
