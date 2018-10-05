# LanguageIdentification

This Github repository contains the code for two language identification tasks, namely monolingual and multilingual language identification. The goal of the models is to identify the predominant language of a given test with a neural network, but where the baseline is not based on a neural network approach, but a simple distance measure. 

## Prerequisites

- Numpy
- Sklearn
- tqdm
- re
- Python modules (collections, time, itertools, operator)
- PyTorch
- Torchtext

## Materials used

- We trained our ```LanguageIdentifier``` models on the WiLI benchmark dataset. (https://arxiv.org/pdf/1801.07779.pdf)
- We trained our ```CodeSwitching``` models on the ALTW2010 Shared Task Dataset. (http://www.aclweb.org/anthology/U10-1003)
- The configuration files used are listed in the ```Configs``` folder.
- Some pretrained models are provided in the ```Models``` folder.

NB: the datasets are slit in training, validation and testing sets. To obtain these splits one can either download the data from the links and use the provided python file for splitting it or open an issue on this repository.


## Training a model

### Baseline
Training and testing are both done after each other with one command for both monolingual and multilingual language identification. For those usages, we refer you to the next section for testing a model. 

### Neural models

Neural models for monolingual language identification (```LanguageIdentifier``` folder) and multilingual language identification (```CodeSwitching```) can be trained in the following way:

```python -m LanguageIdentifier --config config.yaml```

```python -m CodeSwitching --config config.yaml```

Please enter all non-default settings in a configuration file.
Notice that ```LanguageIdentifier``` uses the performance measure accuracy, while ```CodeSwitching``` will provide three measures: micro-averaged and macro-averaged F-scores for language identification and accuracy for the task of text partitioning.

## Testing a model

### Baseline
To train and test for the monolingual and multilingual language identification task, with the same datasets as the neural networks, respectively, the following commands are used:

```python baseline_2.py```

```python baseline_codeSwitch.py```

All settings can be entered at the start of the corresponding python file. _range_val_ can be used to indicate on how many characters the test data will be tested. For the monolingual task, accuracy is used, while for the multilingual task, the f1 score is used during experimenting.

### Neural models

Testing a model can be achieved in the following way:

```python -m LanguageIdentifier --config config.yaml --mode test --max_chars_test 1000 --from_file model.pt```

```python -m CodeSwitching --config config.yaml --mode test --max_chars_test 1000 --from_file model.pt```

The ```max_chars_test``` setting allows you to use a different length cutoff for the testing data compared to the training data, thus resulting in more accurate predictions while testing.

Additionally, ```CodeSwitching``` can be run in inference mode, by simple running:

```python -m CodeSwitching --config config.yaml --mode predict```

