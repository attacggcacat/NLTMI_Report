# from other python file
from pipeline import FF, read_corpus
from collections import Counter

# for the pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer

# visualisation tools
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

# general imports
import numpy as np

# for storing model
import pickle

# create dataloaders
train_data, dev_data = read_corpus('train', 0.75)
test_data, _ = read_corpus('test')

with open("unigram_logreg.pkl", 'rb') as file:
    clf = pickle.load(file)

with open("unigram_dict_vectorizer.pkl", 'rb') as file:
    dict = pickle.load(file)

with open("unigram_tfidf.pkl", 'rb') as file:
    tfdidf = pickle.load(file)

text_clf = Pipeline(
    [
        ('ff', FF(
            lowercase=True,
            byte_unigrams=True,
            )
        ),
        ('dict', dict),   # This will convert python dicts into efficient sparse data structures
        ('tfidf', tfdidf),
        ('clf', clf),
    ]
)

print(classification_report(dev_data[:,1], text_clf.predict(dev_data[:, 0])))
