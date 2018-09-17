import torch
import os
import sys
import numpy as np

from torchtext.data import Iterator
from LanguageIdentifier.model import Model
from LanguageIdentifier.test import test


def train(model : Model, training_data : Iterator, testing_data : Iterator,
          learning_rate : float, epochs : int):

    # NLLLoss for using the log_softmax in the recurrent model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.NLLLoss()

    for i in range(epochs):
        model.train()
        epoch_losses = []
        for j, batch in enumerate(iter(training_data)):

            print("Epoch: {} / Batch: {}".format(i, j), end='\r')
            sys.stdout.flush()

            optimizer.zero_grad()

            # We take the characters as input to the network, and the languages
            # as targets
            characters_reshaped = batch.characters[0]
            characters = torch.autograd.Variable(characters_reshaped)
            languages = batch.language
            predictions = model.forward(characters)
            loss = loss_function(predictions, languages.squeeze(1))
            #epoch_losses.append(loss.item()) 

            # Update the weights
            loss.backward()
            optimizer.step()

        train_accuracy = test(model, training_data)
        test_accuracy = test(model, testing_data)
        print(test_accuracy)
        print("Epoch: {} | Average loss: {} | Train accuracy: {} | Test accuracy: {}".format(
              i + 1, np.mean(np.array(epoch_losses)), train_accuracy, test_accuracy
        ))

