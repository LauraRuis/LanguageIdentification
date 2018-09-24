import torch
import os
import sys
import numpy

from torchtext.data import Iterator
from LanguageIdentifier.model import Model, RecurrentModel


def test(model : Model, testing_data : Iterator) -> float:

    model.eval()
    batch_accuracies = []

    for j, batch in enumerate(iter(testing_data)):
        characters = batch.characters[0]
        languages = batch.language
        if isinstance(model, RecurrentModel):
            predictions = model.forward(characters, batch.characters[1])
        else:
            print("Time: {}".format(characters.shape[1]))
            predictions = model.forward(characters)

        _, predicted_languages = torch.topk(predictions, 1)

        # Save data needed to calculate accuracy for later
        batch_accuracies.extend(languages.eq(predicted_languages))
    return numpy.array([sample.item() for sample in batch_accuracies]).mean()
