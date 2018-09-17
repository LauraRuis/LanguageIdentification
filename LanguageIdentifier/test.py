import torch
import os
import sys
import numpy

from torchtext.data import Iterator
from LanguageIdentifier.model import Model


def test(model : Model, testing_data : Iterator) -> float:

    model.eval()
    batch_accuracies = []

    for j, batch in enumerate(iter(testing_data)):
        # Show progress on test data
        # print("Test batch: {}          ".format(j), end='\r')
        # sys.stdout.flush()
        characters = torch.autograd.Variable(batch.characters[0])
        languages = batch.language
        predictions = model.forward(characters)
        _, predicted_languages = torch.topk(predictions, 1)

        # Save data needed to calculate accuracy for later
        batch_accuracies.extend(languages.eq(predicted_languages))
    return numpy.array([sample.item() for sample in batch_accuracies]).mean()
