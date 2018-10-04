# LanguageIdentification

TODO: intro, tell about language identification
Identify the predominant language of a given text with a neural network.


## Prerequisites


TODO: add what to install


## Training a model

### Baseline

TODO: add how to run the baseline for monolingual and multilingual language identification.

### Neural models

Neural models for monolingual language identification (```LanguageIdentifier``` folder) and multilingual language identification (```CodeSwitching```) can be trained in the following way:

```python -m LanguageIdentifier --config config.yaml```

```python -m CodeSwitching --config config.yaml```

Please enter all non-default settings in a configuration file.
Notice that ```LanguageIdentifier``` uses the performance measure accuracy, while ```CodeSwitching``` will provide three measures: micro-averaged and macro-averaged F-scores for language identification and accuracy for the task of text partitioning.

## Testing a model

### Baseline

### Neural models

Testing a models can be achieved in the following way:

```python -m LanguageIdentifier --config config.yaml --mode test --max_chars_test 1000```

```python -m CodeSwitching --config config.yaml --mode test --max_chars_test 1000```

The ```max_chars_test``` setting allows you to use a different length cutoff for the testing data compared to the training data, thus resulting in more accurate predictions while testing.

Additionally, ```CodeSwitching``` can be run in inference mode, by simple running:

```python -m CodeSwitching --config config.yaml --mode predict```


