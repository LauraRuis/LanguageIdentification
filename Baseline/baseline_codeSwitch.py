import numpy as np
from itertools import groupby
from operator import itemgetter
from collections import Counter
import time
from tqdm import tqdm
from sklearn.metrics import f1_score
import re



train_y_file = "CodeSwitching/trn_lang_labels.txt"
train_x_file = "CodeSwitching/trn_sentences.txt"
train_switch_file = "CodeSwitching/trn_switch_labels.txt"
test_y_file = "CodeSwitching/tst_lang_labels.txt"
test_x_file = "CodeSwitching/tst_sentences.txt"
test_switch_file = "CodeSwitching/tst_switch_labels.txt"
#train_x_file = "wili-2018/x_train.txt"
#train_y_file = "wili-2018/y_train.txt"
#test_x_file = "wili-2018/x_test.txt"
#test_y_file = "wili-2018/y_test.txt"
max_ngram = 5
#k = 20000
k = 3000
penalty = k + 1


def read_data(file_name):
    with open(file_name, 'rb') as f:
        content = f.read().decode("UTF-8")
    return content

def n_grams(input_list, n):
  return list(zip(*[input_list[i:] for i in range(n)]))


def train():
    x_train = read_data(train_x_file)
    x_switch = read_data(train_switch_file)
    x_labels = read_data(train_y_file)
    chars_to_remove = "\(|_|<|\^|@|\\|\&|-|\]|>|\||\+|'|\)|%|\{|\"|\#|\*|\[|\$|/|=|~|`|}|;"
    reg = re.compile(chars_to_remove)
    x_train_text = x_train.split("\r\n")[:-1]
    x_train_text = [re.sub(reg, "", sent) for sent in x_train_text]
    # Not yet ints here
    x_train_switch = [i.split(",") for i in x_switch.split("\r\n")[:-1]]
    x_train_labels = [i.split(",") for i in x_labels.split("\r\n")[:-1]]

    print("Generating data per language...")
    # Maybe also remove punctuation/digits here later??
    # Categorize per language
    combined_data = []
    for i in range(len(x_train_labels)):
        sent = x_train_text[i]
        labels = x_train_labels[i]
        switch_point = int(x_train_switch[i][0])
        first_part = sent[:switch_point-2]
        second_part = sent[switch_point:]
        combined_data.append((first_part, labels[0]))
        combined_data.append((second_part, labels[1]))

    # # Group by every language, first sort
    combined_data.sort(key=lambda x: x[1])
    data_per_language = [list(group) for key, group in groupby(combined_data, itemgetter(1))]
    input_data_per_lang = [([entry[0].lower() for entry in lang_data], lang_data[0][1]) for lang_data in data_per_language]

    for entry, lang in input_data_per_lang:
        print(lang, sum([len(ii) for ii in entry]))


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
        for i in range(min(k, len(language_counts))):
            lang_ranks[language_counts[i][0]] = i + 1
        language_ranks[language] = lang_ranks
        print("One language finished...")

    print("Now going to test data...")

    x_test = read_data(test_x_file)
    test_labels = read_data(test_y_file)
    x_test_text = x_test.split("\r\n")[:-1]
    x_test_text = [re.sub(reg, "", sent).lower() for sent in x_test_text]
    x_test_labels = [i.split(",") for i in test_labels.split("\r\n")[:-1]]

    languages = list(language_ranks.keys())

    predicted_labels = []
    f1_total = 0
    for i in tqdm(range(len(x_test_text))):
        correct_labels = x_test_labels[i]
        x_input = x_test_text[i]
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
        max_val = max(scores, key=itemgetter(1))[1]
        scores = [l for l, sc in scores if sc/max_val > 0.8]
        scores.sort(key=lambda tup: tup[1], reverse=True)
        predicted_labels.append(scores)
        f1_score = f1(scores, correct_labels)
        f1_total += f1_score
        if (i % 500) == 0:
            print(f1_total/(i+1))
    f1_total = f1_total/len(x_test_text)
    print("F1: " + str(f1_total))
    with open("predicted_300_codeswitch.txt", 'w') as f:
        for item in predicted_labels:
            f.write("%s\n" % item)

def f1(scores, correct_labels):
    relevant_retrieved = len(set(scores).intersection(set(correct_labels)))
    precision = relevant_retrieved/len(scores)
    recall = relevant_retrieved/len(correct_labels)
    if (precision == 0) and (recall == 0):
        return 0
    return 2 * ((precision * recall)/(precision + recall))

if __name__ == "__main__":
    train()
