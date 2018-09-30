import torch
import os
import sys
import numpy

from torchtext.data import Iterator
from LanguageIdentifier.model import Model, RecurrentModel


def test(model : Model, testing_data : Iterator, output_matrix : bool=False) -> float:

    model.eval()
    batch_accuracies = []
    classes = testing_data.dataset.fields['language'].vocab.itos
    n_classes = len(classes)
    confusion_matrix = numpy.zeros((n_classes, n_classes))

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

        for p, t in zip(predicted_languages, languages):
            if p != t:
                confusion_matrix[p][t] += 1

    if output_matrix:
        with open("confusion_matrix.txt", 'w') as f:
            f.write("\t")
            f.write("\t".join(classes))
            f.write("\n")
            for i, line in enumerate(confusion_matrix):
                f.write("{}\t".format(classes[i]))
                f.write("\t".join(map(str, line)))
                f.write("\n")

    return numpy.array([sample.item() for sample in batch_accuracies]).mean()
