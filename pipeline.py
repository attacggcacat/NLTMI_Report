import os.path as path
import random
from collections import Counter
from tempfile import mkdtemp

import numpy as np
# import torch
import pickle
from sklearn.base import TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# from torch.autograd import Variable
# from torch.nn import functional as F
# from torch.utils.data import DataLoader

SEED = 42

# heavily inspired by assignment 2's prepare_corpus


def prepare_corpus(mode='train', train_partition=1., n_samples=117500):
    filename = path.join(mkdtemp(), 'array.dat')
    fp = np.memmap(filename, dtype='object', mode='w+', shape=(n_samples, 2))

    with open(f'x_{mode}.txt', 'r') as f_doc_examples, \
            open(f'y_{mode}.txt', 'r') as f_doc_labels:

        for i in range(n_samples):
            example = f_doc_examples.readline().rstrip('\n')
            label = f_doc_labels.readline().rstrip('\n')
            fp[i, ] = (example, label)

    rng = np.random.RandomState(SEED)
    rng.shuffle(fp)
    fp.flush()

    if mode == 'train':
        train_filename = path.join(mkdtemp(), 'train.dat')
        training = np.memmap(
            train_filename, dtype='object', mode='w+',
            shape=(int(train_partition * n_samples), 2)
        )
        training = fp[:int(n_samples * train_partition)]
        training.flush()

        dev_filename = path.join(mkdtemp(), 'dev.dat')
        dev = np.memmap(
            dev_filename, dtype='object', mode='w+',
            shape=(n_samples - int(train_partition * n_samples), 2)
        )
        dev = fp[int(n_samples * train_partition):]
        dev.flush

        return training, dev

    else:
        return filename


# from assignment 2
class FF(TransformerMixin):
    """
    Our input is text
    which we will transform to a dictionary of sparse features (using python dict).

    Check the example and then include your own ideas for features.
    """

    def __init__(self, lowercase=False, byte_unigrams=False, byte_bigrams=False):
        """
        :param lowercase: should we lowercase before doing anything?
        :param unigrams: count characters
        """
        self._lowercase = lowercase
        self._byte_unigrams = byte_unigrams

    def _ff(self, string):
        """
        This is our feature function, it maps a document to a space of real-valued features.
        It uses python dict to represent only the features with non-zero values.

        :param string: a document as a string
        "returns: dict (key is string, value is int/float)
        """
        fvec = Counter()

        string.encode('utf-8')

        # we can do some pre-processing if we like
        if self._lowercase:
            string = string.lower()

        if self._byte_unigrams:
            fvec.update(('unigram={}'.format(byte) for byte in string))

        # if self._byte_bigrams:
        #     for byte in
        #     fvec.update(('bigram={}'.format(w) for w in string))

        return fvec

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return dict()

    def transform(self, X, **transform_params):
        """Here we transform each input (a string) into a python dict full of features"""
        return [self._ff(s) for s in X]


if __name__ == "__main__":
    text_log_clf = Pipeline(
        [
            ('ff', FF(
                lowercase=True,
                byte_unigrams=True,
            )
            ),
            # This will convert python dicts into efficient sparse data structures
            ('dict', DictVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', LogisticRegression(max_iter=500, verbose=2, C=100., solver='sag')),
        ]
    )

    text_nbc_clf = Pipeline(
        [
            ('ff', FF(
                lowercase=True,
                byte_unigrams=True,
            )
            ),
            # This will convert python dicts into efficient sparse data structures
            ('dict', DictVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('NBC', MultinomialNB(alpha=0.5)),
        ]
    )
    # create dataloaders
    print('preparing corpus')
    n_samples = 117500
    train_partition = 0.75
    n_train_samples = int(train_partition * n_samples)

    train_data, dev_data = prepare_corpus(
        'train', train_partition, n_samples=n_samples)

    print(train_data[:1,])
    text_log_clf.fit(train_data[:, 0], train_data[:, 1])
    with open("unigram_dict_vectorizer.pkl", 'wb') as f:
        pickle.dump(text_log_clf['dict'], f)

    with open("unigram_tfidf.pkl", 'wb') as f:
        pickle.dump(text_log_clf['tfidf'], f)

    with open("unigram_logreg.pkl", 'wb') as f:
        pickle.dump(text_log_clf['clf'], f)
        
    print(classification_report(
        dev_data[:, 1], text_log_clf.predict(dev_data[:, 0]),
        zero_division=0)
    )
