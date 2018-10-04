from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
from operator import itemgetter

predictions = "predicted_1024.txt"
# predictions = "predicted_10000.txt"
true_labels = "wili-2018/y_test.txt"

def read_data(file_name):
    with open(file_name, 'rb') as f:
        content = f.read().decode("UTF-8")
    return content

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def make_confusion_matrix():
    prediction_list = read_data(predictions).split("\n")[:-1]
    true_list = read_data(true_labels).split("\n")[:-1]
    class_names = list(set(true_list))
    matrix = confusion_matrix(true_list, prediction_list, class_names)
    confusions = []
    for class_id in range(len(class_names)):
        for class_id_2 in range(len(class_names)):
            if class_id == class_id_2:
                print("Correct: " + class_names[class_id])
            else:
                val = matrix[class_id][class_id_2]
                if val != 0:
                    print("Incorrect:" + class_names[class_id_2], "Value: " + str(val))
                    confusions.append(((class_names[class_id], class_names[class_id_2]), val))
        print("----------------------------------------------------------")
    confusions.sort(key=lambda x: x[1], reverse=True)
    print(confusions[:20])

    #plt.figure()
    #plot_confusion_matrix(matrix, classes=class_names,
    #                  title='Confusion matrix, without normalization')
    plt.show()
if __name__ == "__main__":
    make_confusion_matrix()
