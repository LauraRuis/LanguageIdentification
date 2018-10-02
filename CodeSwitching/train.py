import torch
import os
import sys
import numpy as np
import math
import datetime

from torchtext.data import Iterator
from torch.optim import adam

from CodeSwitching.model import Model, RecurrentModel, GRUIdentifier
from CodeSwitching.test import test, calculate_accuracy
from CodeSwitching.utils import save_model, PAD_TOKEN


def train(optimizer: adam=None, model: Model=None,
          training_data: Iterator=None, validation_data: Iterator=None, testing_data: Iterator=None,
          learning_rate: float=1e-3, epochs: int=0, resume_state: dict=None, resume: str="",
          log_frequency: int=0, eval_frequency: int=0, model_type="", output_dir="",
          level : str="char",
          **kwargs):

    # get command line arguments
    cfg = locals().copy()

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if not resume:
        best_train_acc, best_val_acc, best_test_acc = 0, 0, 0
        start_epoch = 0
    else:
        start_epoch = resume_state['epoch']
        training_acc = resume_state['train_acc']
        validation_acc = resume_state['val_acc']
        test_acc = resume_state['test_acc']
        best_val_acc = validation_acc

    print(datetime.datetime.now(), " Training at {} level starts.".format(level))
    batch_accuracies, epoch_accuracies = {"TP" : [], "CL" : []}, {"TP" : [], "CL" : []}
    for i in range(start_epoch, epochs):
        epoch_losses = []

        for j, batch in enumerate(iter(training_data)):
            model.train()
            optimizer.zero_grad()

            # We take the characters as input to the network, and the languages
            # as targets
            if level == 'char':
                sequence = batch.characters[0]
                lengths = batch.characters[1]
                target = batch.language_per_char[0]
                # NLLLoss for using the log_softmax in the recurrent model
                pad_idx = training_data.dataset.fields['characters'].vocab.stoi[PAD_TOKEN]
                loss_function = torch.nn.NLLLoss(ignore_index=pad_idx)
            else:
                sequence = batch.paragraph[0]
                lengths = batch.paragraph[1]
                char_lengths = batch.paragraph[2]
                target = batch.language_per_word[0]
                # NLLLoss for using the log_softmax in the recurrent model
                pad_idx = training_data.dataset.fields['paragraph'].vocab.stoi[PAD_TOKEN]
                loss_function = torch.nn.NLLLoss(ignore_index=pad_idx)

            batch_size = sequence.shape[0]

            if isinstance(model, RecurrentModel):
                predictions = model.forward(sequence, lengths)
            else:
                predictions = model.forward(sequence, char_lengths, lengths)

            bsz, time, n_languages = predictions.shape
            loss = loss_function(predictions.view(bsz * time, n_languages), target.view(-1))
            epoch_losses.append(loss.item())

            _, predicted_languages = torch.topk(predictions, 1)

            # Save data needed to calculate accuracy for later
            batch_accuracy_tp, batch_accuracy_cl = calculate_accuracy(predictions, target, lengths)
            batch_accuracies["TP"].append(batch_accuracy_tp)
            batch_accuracies["CL"].append(batch_accuracy_cl)
            epoch_accuracies["TP"].append(batch_accuracy_tp)
            epoch_accuracies["CL"].append(batch_accuracy_cl)


            # Update the weights
            loss.backward()
            optimizer.step()

            if (j + 1) % cfg["log_frequency"] == 0:
                print(datetime.datetime.now(), " Logging: Epoch: {} | Iter: {} | Loss: {:.4f} | Batch acc. CL: {:.2f} | Batch acc. TP: {:.2f}".format(
                    i, j, loss.item(), batch_accuracies["CL"][-1], batch_accuracies["TP"][-1]))

            if (j + 1) % cfg["eval_frequency"] == 0:
                train_accuracy_cl = np.array(batch_accuracies["CL"]).mean()
                train_accuracy_tp = np.array(batch_accuracies["TP"]).mean()

                batch_accuracies = {"TP" : [], "CL" : []}
                validation_accuracy_tp, validation_accuracy_cl = test(model, validation_data, level)
                print(datetime.datetime.now(), " Evaluation: Epoch: {} | Iter: {} | Loss: {} | "
                      "Av. Batch Train acc. CL {:.4f}| Batch Train acc. TP {:.4f}| | Validation acc. CL {:.2f} | Validation acc. TP {:.2f}".format(
                      i, j, loss.item(), train_accuracy_cl, train_accuracy_tp, validation_accuracy_cl, validation_accuracy_tp
                ))
                if validation_accuracy_cl > best_val_acc:
                    best_train_acc_cl = train_accuracy_cl
                    best_train_acc_tp = train_accuracy_tp
                    best_val_acc = validation_accuracy_cl
                    output_dir = cfg["output_dir"]
                    save_model(output_dir,
                               {
                                   'epoch': i,
                                   'state_dict': model.state_dict(),
                                   'train_acc_cl': train_accuracy_cl,
                                   'val_acc_cl': validation_accuracy_cl,
                                   'train_acc_tp': train_accuracy_tp,
                                   'val_acc_tp': validation_accuracy_tp,
                                   'optimizer': optimizer.state_dict(),
                               },
                               filename=cfg["model_type"] + "_best_model.pth.tar")
                    torch.save(model, output_dir + "/" + cfg["model_type"] + "_best_model.pt")

        train_accuracy_tp = np.array(epoch_accuracies["TP"]).mean()
        train_accuracy_cl = np.array(epoch_accuracies["CL"]).mean()

        epoch_accuracies = {"TP" : [], "CL" : []}
        validation_accuracy_tp, validation_accuracy_cl = test(model, validation_data, level)
        test_accuracy_tp, test_accuracy_cl = test(model, testing_data, level)

        print(datetime.datetime.now(), " Epoch: {} finished | Average loss: {:.4f} | "
              "Av. Batch Train accuracy CL: {:.2f} | Av. Batch Train accuracy TP: {:.2f} | Validation acc. CL {:.2f} | Validation acc. TP {:.2f}".format(
              i + 1, np.mean(np.array(epoch_losses)), train_accuracy_cl, train_accuracy_tp, validation_accuracy_cl, validation_accuracy_tp
        ))

    print(datetime.datetime.now(), " Done training.")
    print("Best model: Train accuracy CL: {:.2f} | Train accuracy TP: {:.2f} | Validation acc. CL {:.2f} | Validation acc. TP {:.2f} \
          | Test acc. CL {:.2f} | Test acc. TP {:.2f} | ".format(
          best_train_acc_cl, best_train_acc_tp, validation_accuracy_cl, validation_accuracy_tp, test_accuracy_cl, test_accuracy_tp
    ))
