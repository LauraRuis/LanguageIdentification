import torch
import os
import sys
import numpy
from collections import Counter, defaultdict

from torchtext.data import Iterator
from LanguageIdentifier.model import Model, RecurrentModel, GRUIdentifier


THRESHOLD = 0


def print_example(sequence, prediction, target, data):
    itos = data.dataset.fields['paragraph'].vocab.itos
    words = [itos[i] for i in sequence]
    itos = data.dataset.fields['language_per_word'].vocab.itos
    prediction = [itos[torch.argmax(i).item()] for i in prediction]
    target = [itos[i.item()] for i in target]
    print(words, prediction, target)


def calculate_scores(scores, prediction, target):
    for language in prediction:
        if language in target:
            scores[language]["TP"] += 1
        else:
            scores[language]["FP"] += 1

    for language in target:
        if language not in prediction:
            scores[language]["FN"] += 1
    return scores


def to_languages(languages):
    language_count = Counter(languages)
    return set(lan for lan in languages if language_count[lan] > THRESHOLD)


def calculate_accuracy(predictions : torch.Tensor, targets : torch.Tensor,
                       lengths : torch.Tensor, n_classes : int,
                       scores : Counter=None,
                       confusion_matrix : numpy.matrix=None,
                       sparse_matrix : Counter=None) -> float:

    accuracies = Counter()
    
    if scores is None:
        scores = defaultdict(lambda : { 'TP' : 0, 'FP' : 0, 'FN' : 0})

    for prediction, target, length in zip(predictions, targets, lengths):
        predicted_labels = torch.argmax(prediction, 1)[:(length-1)]
        target = target[:(length-1)]

        # Text partitioning: all labels should be correct
        if torch.sum(predicted_labels.eq(target)) == len(target):
            accuracies["text_partitioning"] += 1.0

        # Only output confusion matrix in test mode
        if confusion_matrix is not None:
            for p, t in zip(predicted_labels, target):
                if p != t: 
                    confusion_matrix[p][t] += 1
                    sparse_matrix[(p,t)] += 1

        predicted_labels = to_languages(predicted_labels.cpu().numpy()) 
        target = set(target.cpu().numpy())
        scores = calculate_scores(scores, predicted_labels, target)

        # Classification: only the set of labels should be correct
        if set(predicted_labels) == set(target):
            accuracies["classification"] += 1.0

    acc_tp = accuracies["text_partitioning"] / lengths.shape[0]
    acc_cl = accuracies["classification"] / lengths.shape[0]
    F_micro, _ = to_f_score(scores)

    if confusion_matrix is not None and scores is not None:
        return acc_tp, acc_cl, confusion_matrix, sparse_matrix, scores
    else:
        return acc_tp, F_micro


def test(model : Model, testing_data : Iterator, level : str, show_example : bool=False) -> float:

    model.eval()
    batch_accuracies = {"text_partitioning" : [], "classification" : []}
    if level == "char":
        classes = testing_data.dataset.fields['language_per_char'].vocab.itos
    else:
        classes = testing_data.dataset.fields['language_per_word'].vocab.itos
    n_classes = len(classes)
    confusion_matrix = numpy.zeros((n_classes, n_classes))
    sparse_matrix = Counter()
    scores = defaultdict(lambda : { 'TP' : 0, 'FP' : 0, 'FN' : 0})


    for j, batch in enumerate(iter(testing_data)):

        if level == 'char':
            sequence = batch.characters[0]
            lengths = batch.characters[1]
            target = batch.language_per_char[0]
        else:
            sequence = batch.paragraph[0]
            lengths = batch.paragraph[1]
            char_lengths = batch.paragraph[2]
            target = batch.language_per_word[0]

        if model.name == "recurrent":            
            predictions = model.forward(sequence, lengths)
        else:
            predictions = model.forward(sequence, char_lengths, lengths)

        # Save data needed to calculate accuracy for later
        tp_accuracy, cl_accuracy, confusion_matrix, sparse_matrix, scores = \
            calculate_accuracy(
                predictions, target, lengths, n_classes, scores, 
                confusion_matrix, sparse_matrix
            )
        batch_accuracies["text_partitioning"].append(tp_accuracy)
        batch_accuracies["classification"].append(cl_accuracy)

        # if show_example:
        #     print_example(sequence[0, :], predictions[0, :], target[0, :], testing_data)

    # Write matrices to file
    if show_example:
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
                lan = (classes[lan[0].cpu().item()], classes[lan[1].cpu().item()])
                f.write("{}: {}\n".format(lan, score))

    F_micro, F_macro = to_f_score(scores)

    acc_tp = numpy.array([acc for acc in batch_accuracies["text_partitioning"]]).mean()
    acc_cl = numpy.array([acc for acc in batch_accuracies["classification"]]).mean()
    return acc_tp, F_micro, F_macro

def to_f_score(scores):
    # micro-averaged
    true_positives = sum(scores[lan]['TP'] for lan in scores)
    false_positives = sum(scores[lan]['FP'] for lan in scores)
    false_negatives = sum(scores[lan]['FN'] for lan in scores)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    if (precision + recall) > 0:
        F_micro = 2 * ( precision * recall ) / (precision + recall)
    else:
        F_micro = 0

    # macro-averaged
    precision, recall = 0, 0
    for lan in scores:
        denom = (scores[lan]['TP'] + scores[lan]['FP'])
        if denom > 0: precision += scores[lan]['TP'] / denom
        denom = scores[lan]['TP'] + scores[lan]['FN']
        if denom > 0: recall += scores[lan]['TP'] / denom
    precision = precision / len(scores)
    recall = recall / len(scores)
    F_macro = 2 * ( precision * recall ) / (precision + recall)
    return F_micro, F_macro