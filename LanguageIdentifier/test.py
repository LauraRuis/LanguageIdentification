import torch
import os
import sys
import numpy

from collections import Counter
from torchtext.data import Iterator
from LanguageIdentifier.model import Model, RecurrentModel, CharModel
from LanguageIdentifier.utils import PAD_TOKEN


def test(model : Model, testing_data : Iterator, output_matrix : bool=False, level: str='char') -> float:

    model.eval()
    batch_accuracies = []
    classes = testing_data.dataset.fields['language'].vocab.itos
    n_classes = len(classes)
    confusion_matrix = numpy.zeros((n_classes, n_classes))
    sparse_matrix = Counter()

    for j, batch in enumerate(iter(testing_data)):

        if level == 'char':
            sequence = batch.characters[0]
            lengths = batch.characters[1]
            target = batch.language
        else:
            sequence = batch.paragraph[0]
            lengths = batch.paragraph[1]
            char_lengths = batch.paragraph[2]
            target = batch.language

        if isinstance(model, RecurrentModel):
            predictions = model.forward(sequence, batch.characters[1])
        elif isinstance(model, CharModel):
            predictions = model.forward(sequence)
        else:
            predictions = model.forward(sequence, char_lengths, lengths)

        _, predicted_languages = torch.topk(predictions, 1)

        # Save data needed to calculate accuracy for later
        batch_accuracies.extend(target.eq(predicted_languages))

        for p, t in zip(predicted_languages, target):
            if p != t:
                confusion_matrix[p][t] += 1
                sparse_matrix[(classes[p],classes[t])] += 1

    if output_matrix:
        with open("confusion_matrix.txt", 'w') as f:
            f.write("\t")
            f.write("\t".join(classes))
            f.write("\n")
            for i, line in enumerate(confusion_matrix):
                f.write("{}\t".format(classes[i]))
                f.write("\t".join(map(str, line)))
                f.write("\n")

        with open("sparse_matrix.txt", 'w') as f:
            for lan, score in sparse_matrix.most_common():
                f.write("{} - {} : {}\n".format(lan[0], lan[1], score))

    return numpy.array([sample.item() for sample in batch_accuracies]).mean()
