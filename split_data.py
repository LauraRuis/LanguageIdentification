from collections import Counter

or_test_data = []
test_file = open('./Data/x_test.txt', 'r') 
for line in test_file: 
    or_test_data.append(line)
test_file.close()

or_test_labels = [] 
test_lab_file = open('./Data/y_test.txt', 'r') 
for line in test_lab_file: 
    or_test_labels.append(line)
test_lab_file.close()

train_data = []
train_file = open('./Data/x_train.txt', 'r') 
for line in train_file: 
    train_data.append(line)
train_file.close()

train_labels = [] 
train_lab_file = open('./Data/y_train.txt', 'r') 
for line in train_lab_file: 
    train_labels.append(line)
train_lab_file.close()

c_train = Counter(train_labels)
c_test = Counter(or_test_labels)

all_labels = []
valid_data = []
valid_labels = []
test_data = []
test_labels = []

for i in range(len(or_test_data)):
    all_labels.append(or_test_labels[i])
    c = Counter(all_labels)
    word_count = c[or_test_labels[i]]
    if word_count <= 300:
        train_data.append(or_test_data[i])
        train_labels.append(or_test_labels[i])
    elif 300 < word_count <= 400:
        valid_data.append(or_test_data[i])
        valid_labels.append(or_test_labels[i])
    elif word_count > 400:
        test_data.append(or_test_data[i])
        test_labels.append(or_test_labels[i])
    else:
        print(' not in the previous options, count', word_count)


print('\nnumber of lines in training set:   ', len(train_data))

print('\nnumber of lines in validation set: ', len(valid_data))

print('\nnumber of lines in test set:       ', len(test_data))

# write new training files
with open('./Data/x_train_split.txt', 'w') as f:
    for item in train_data:
        f.write("%s" % item)
f.close()

with open('./Data/y_train_split.txt', 'w') as f:
    for item in train_labels:
        f.write("%s" % item)
f.close()

# write new validation files
with open('./Data/x_valid_split.txt', 'w') as f:
    for item in valid_data:
        f.write("%s" % item)
f.close()

with open('./Data/y_valid_split.txt', 'w') as f:
    for item in valid_labels:
        f.write("%s" % item)
f.close()

# write new test files
with open('./Data/x_test_split.txt', 'w') as f:
    for item in test_data:
        f.write("%s" % item)
f.close()

with open('./Data/y_test_split.txt', 'w') as f:
    for item in test_labels:
        f.write("%s" % item)
f.close()