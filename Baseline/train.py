import torch
import torch.nn as nn
import numpy as np
from itertools import groupby
from operator import itemgetter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from model import MLP
from collections import Counter

train_x_file = "wili-2018/x_train.txt"
train_y_file = "wili-2018/y_train.txt"
test_x_file = "wili-2018/x_test.txt"
test_y_file = "wili-2018/y_test.txt"
min_occurences = 100
n_hidden = [100]
batch_size = 235
epochs = 20
learning_rate = 0.01

def read_data(file_name):
    with open(file_name, 'rb') as f:
        content = f.read().decode("UTF-8")
    return content

def accuracy(predictions, targets):
    """
    predictions and labels as 2D float and int array of batch_size x n_classes
    """
    # Obtain the indices of the highest predictions
    _, pred_indices = torch.max(predictions, 1)
    _, target_indices = torch.max(targets, 1)
    # Count how many are correct
    correct_ones = sum(pred_indices == target_indices).item()
    accuracy = correct_ones/len(pred_indices)

    return accuracy

def construct_data(x_file, y_file, train_test, label_encoder=None, label_onehot=None,
                    tfidf_object=None):
    # Fix the data
    x_orig = read_data(x_file).split("\n")[:-1]
    y = read_data(y_file).split("\n")[:-1]

    if train_test == "train":
        label_encoder = LabelEncoder()
        label_encoder.fit(y)
    y_num = label_encoder.transform(y)

    if train_test == "train":
        label_onehot = LabelBinarizer()
        label_onehot.fit(y_num)
    y_onehot = label_onehot.transform(y_num)
    y_batch = [torch.Tensor(y_onehot[i:i + batch_size]) for i in range(0, len(y_onehot), batch_size)]

    if train_test == "train":
        # First do something with occurences less than 100, make them L
        all_words = []
        for sent in x_orig:
            all_words.extend(list(sent))
        occs = list(Counter(all_words).most_common())
        print(len(occs))
        occs.reverse()
        chars_to_replace = []
        for char, num in occs:
            if num < 100:
                chars_to_replace.append(char)
            else:
                break

        # Now clean out the data with a copy!
        x = [xi[:100] for xi in read_data(x_file).split("\n")[:-1]]
        for i in range(len(x_orig)):
            for char in chars_to_replace:
                x[i].lower().replace(char, "O")

        combined_data = list(zip(x, y))
        # Group by every language, first sort
        combined_data.sort(key=lambda x: x[1])
        data_per_language = [list(group) for key, group in groupby(combined_data, itemgetter(1))]
        input_data_per_lang = ["".join([entry[0][:100] for entry in lang_data]) for lang_data in data_per_language]
        tfidf_object = TfidfVectorizer(analyzer="char", lowercase=True, min_df=100,
                                        norm="l2").fit(input_data_per_lang)
        print(tfidf_object.get_feature_names())
        print(len(tfidf_object.get_feature_names()))
        tfidf_object.fit(x)
    #else:
        # Now clean out the data with a copy!
        #x = read_data(x_file).split("\n")[:-1]
        #for i in range(len(x_orig)):
        #    for char in chars_to_replace:
        #        x[i].lower().replace(char, "O")


    x_feats = tfidf_object.transform(x)
    x_feats = x_feats.toarray()
    x_feats_batch = [torch.Tensor(x_feats[i:i + batch_size]) for i in range(0, len(x_feats), batch_size)]
    return x_feats_batch, y_batch, label_encoder, label_onehot, tfidf_object

def train():
    print("Obtaining the training data...")
    train_x_feats_batch, train_y_batch, label_encoder, label_onehot, tfidf_object = construct_data(train_x_file,
                                            train_y_file, "train")
    print("Obtaining the test data...")
    test_x_feats_batch, test_y_batch, _, _, _= construct_data(test_x_file, test_y_file, "test", label_encoder,
                                            label_onehot, tfidf_object)

    print("Set values....")
    n_inputs = train_x_feats_batch[0].shape[1]
    n_classes = train_y_batch[0].shape[1]
    model = MLP(n_inputs, n_hidden, n_classes)

    loss_function = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Start training...")
    for epoch in range(epochs):
        print("Doing epoch: " + str(epoch + 1))
        for batch_num in range(len(train_x_feats_batch)):
            optimizer.zero_grad()
            batch_x = train_x_feats_batch[batch_num]
            batch_y = train_y_batch[batch_num]

            output = model(batch_x)
            _, targets = torch.max(batch_y.long(), 1)
            found_loss = loss_function(output, targets)
            found_loss.backward()
            optimizer.step()
        avg_acc = 0
        print("Now evaluating..")
        for batch_num in range(len(test_x_feats_batch)):
            batch_x = test_x_feats_batch[batch_num]
            batch_y = test_y_batch[batch_num]
            output = model(batch_x)
            acc = accuracy(output, batch_y)
            avg_acc += acc
        avg_acc = avg_acc/len(test_x_feats_batch)
        print("Current Accuracy: " + str(avg_acc))



if __name__ == "__main__":
    train()
