import numpy as np
from itertools import groupby
from operator import itemgetter
from collections import Counter
import time
from tqdm import tqdm
import re

train_x_file = "wili-2018/x_train.txt"
train_y_file = "wili-2018/y_train.txt"
test_x_file = "wili-2018/x_test.txt"
test_y_file = "wili-2018/y_test.txt"
max_ngram = 5
k = 10000
#k = 300
penalty = k + 1


def read_data(file_name):
    with open(file_name, 'rb') as f:
        content = f.read().decode("UTF-8")
    return content

def n_grams(input_list, n):
  return list(zip(*[input_list[i:] for i in range(n)]))


def train():
    x = read_data(train_x_file).split("\n")[:-1]
    y = read_data(train_y_file).split("\n")[:-1]

    print("Generating data per language...")
    # Maybe also remove punctuation/digits here later??
    # Categorize per language
    combined_data = list(zip(x, y))
    # Group by every language, first sort
    combined_data.sort(key=lambda x: x[1])
    data_per_language = [list(group) for key, group in groupby(combined_data, itemgetter(1))]
    input_data_per_lang = [([entry[0].lower() for entry in lang_data], lang_data[0][1]) for lang_data in data_per_language]

    #for entry, lang in input_data_per_lang:
    #    print(lang, sum([len(ii) for ii in entry]))

    print("Now creating n-grams....")
    language_ranks = {}
    for sents, language in input_data_per_lang:
        n_grams_lang = []
        for sent in sents:
            words = sent.split(" ")
            for word in words:
                n_grams_lang.extend(n_grams(word, 1))
                for n_gram in range(2, max_ngram + 1):
                    n_grams_lang.extend(n_grams(" " + word + "".join([" "] * (n_gram - 1)), n_gram))
        language_counts = list(Counter(n_grams_lang).most_common())
        lang_ranks = {}
        for i in range(k):
            lang_ranks[language_counts[i][0]] = i + 1
        language_ranks[language] = lang_ranks
        print("One language finished...")

    print("Now going to test data...")
    range_val = 512
    x_test = read_data(test_x_file).split("\n")[:-1]
    x_test = [i.lower()[:range_val] for i in x_test]
    print(x_test[0])
    y_test = read_data(test_y_file).split("\n")[:-1]
    languages = list(language_ranks.keys())

    predicted_labels = []
    acc = 0
    for i in tqdm(range(len(x_test))):
        correct_label = y_test[i]
        x_input = x_test[i]
        x_words = x_input.split(" ")
        n_grams_lang = []
        for word in x_words:
            n_grams_lang.extend(n_grams(word, 1))
            for n_gram in range(2, max_ngram + 1):
                n_grams_lang.extend(n_grams(" " + word + "".join([" "] * (n_gram - 1)), n_gram))

        language_counts = list(Counter(n_grams_lang).most_common())
        lang_ranks = {}
        scores = []
        for lang in languages:
            score = 0
            current_model = language_ranks[lang]
            for j in range(min(len(language_counts), k)):
                feature = language_counts[j][0]
                rank_cg = current_model.get(feature)
                if rank_cg:
                    score += abs((j + 1) - rank_cg)
                else:
                    score += penalty
            scores.append((lang, score))
        prediction = min(scores, key=itemgetter(1))[0]
        predicted_labels.append(prediction)
        if prediction == correct_label:
            acc += 1
        if (i % 500) == 0:
            print(acc/(i+1))
    acc = acc/len(x_test)
    print("Accuracy: " + str(acc))
    #with open("predicted_10000.txt", 'w') as f:
    #    for item in predicted_labels:
    #        f.write("%s\n" % item)




if __name__ == "__main__":
    train()
