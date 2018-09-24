import torch
import os
import sys
import numpy

from torchtext.data import Iterator
from LanguageIdentifier.model import Model, RecurrentModel, GRUIdentifier

def print_example(sequence, prediction, target, data):
    itos = data.dataset.fields['paragraph'].vocab.itos
    words = [itos[i] for i in sequence]
    itos = data.dataset.fields['language_per_word'].vocab.itos
    prediction = [itos[torch.argmax(i).item()] for i in prediction]
    target = [itos[i.item()] for i in target]

    print(words, prediction, target)

def calculate_accuracy(predictions : torch.Tensor, targets : torch.Tensor, lengths : torch.Tensor) -> float:
    correct_sequences = 0.0
    all_sequences = 0.0
    for prediction, target, length in zip(predictions, targets, lengths):
        predicted_labels = torch.argmax(prediction, 1)[:(length-1)]
        target = target[:(length-1)]
        if torch.sum(predicted_labels.eq(target)) == len(target):
          correct_sequences += 1.0
        all_sequences += 1.0
    return correct_sequences / all_sequences


def test(model : Model, testing_data : Iterator, level : str, lengths : torch.Tensor, show_example : bool=False) -> float:

    model.eval()
    batch_accuracies = []

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
        batch_accuracies.append(calculate_accuracy(predictions, target, lengths))
        # if show_example:
        #     print_example(sequence[0, :], predictions[0, :], target[0, :], testing_data)
    return numpy.array([acc for acc in batch_accuracies]).mean()
