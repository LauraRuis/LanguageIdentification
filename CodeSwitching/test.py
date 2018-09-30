import torch
import os
import sys
import numpy
from collections import Counter

from torchtext.data import Iterator
from LanguageIdentifier.model import Model, RecurrentModel, GRUIdentifier

def print_example(sequence, prediction, target, data):
    itos = data.dataset.fields['paragraph'].vocab.itos
    words = [itos[i] for i in sequence]
    itos = data.dataset.fields['language_per_word'].vocab.itos
    prediction = [itos[torch.argmax(i).item()] for i in prediction]
    target = [itos[i.item()] for i in target]

    print(words, prediction, target)

def calculate_accuracy(predictions : torch.Tensor, targets : torch.Tensor,
                               lengths : torch.Tensor, confusion_matrix : numpy.matrix=None) -> float:
    accuracies = Counter()
    for prediction, target, length in zip(predictions, targets, lengths):
        predicted_labels = torch.argmax(prediction, 1)[:(length-1)]
        target = target[:(length-1)]
        if torch.sum(predicted_labels.eq(target)) == len(target):
            accuracies["text_partitioning"] += 1.0
        if set(predicted_labels.cpu().numpy()) == set(target.cpu().numpy()):
            accuracies["classification"] += 1.0

        if confusion_matrix is not None:
            for p, t in zip(predicted_labels, target):
                if p != t: confusion_matrix[p][t] += 1
    if confusion_matrix is not None:
        return accuracies["classification"] / lengths.shape[0], confusion_matrix
    else:
        return accuracies["classification"] / lengths.shape[0]


def test(model : Model, testing_data : Iterator, level : str, lengths : torch.Tensor, show_example : bool=False) -> float:

    model.eval()
    batch_accuracies = []
    classes = testing_data.dataset.fields['language_per_char'].vocab.itos
    n_classes = len(classes)
    confusion_matrix = numpy.zeros((n_classes, n_classes))

    for j, batch in enumerate(iter(testing_data)):

        if level == 'char':
            sequence = batch.characters[0]
            lengths = batch.characters[1]
            target = batch.language_per_char[0]
        else:
            sequence = batch.paragraph[0]
            lengths = batch.paragraph[1]
            target = batch.language_per_word[0]

        if model.name == "recurrent":            
            predictions = model.forward(sequence, lengths)
        else:
            predictions = model.forward(sequence)

        # Save data needed to calculate accuracy for later
        accuracy, confusion_matrix = calculate_accuracy(predictions, target, lengths, confusion_matrix)
        batch_accuracies.append(accuracy)
        # if show_example:
        #     print_example(sequence[0, :], predictions[0, :], target[0, :], testing_data)
    
    if show_example:
        with open("confusion_matrix.txt", 'w') as f:
            f.write("\t")
            f.write("\t".join(classes))
            f.write("\n")
            for i, line in enumerate(confusion_matrix):
                f.write("{}\t".format(classes[i]))
                f.write("\t".join(map(str, line)))
                f.write("\n")


    return numpy.array([acc for acc in batch_accuracies]).mean()
