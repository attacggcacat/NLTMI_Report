# for feature function
from sklearn.base import TransformerMixin
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

SEED = 42

# heavily inspired by assignment 2's prepare_corpus
def read_corpus(mode='train', train_partition=1.):

    # read the texts and their labels
    f_doc_examples = open(f'x_{mode}.txt', 'r')
    f_doc_labels = open(f'y_{mode}.txt', 'r')


    # pair language segments with their label
    pairs = []


    # USING SMALL PART OF DATASET
    for i in range(8000):
        example = f_doc_examples.readline()
        label = f_doc_labels.readline()

        # discard last newline character
        example = example[:-1]
        label = label[:-1]

        pairs.append((example, label))

    pairs = np.array(pairs)

    # randomly shuffle pairs
    rng = np.random.RandomState(SEED)
    rng.shuffle(pairs)

    # split into train and devset
    num_pairs = pairs.shape[0]

    training = pairs[0:int(num_pairs * train_partition),:]

    dev = pairs[int(num_pairs * train_partition):,:]

    return training, dev


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

def main():
    text_clf = Pipeline(
        [
            ('ff', FF(
                lowercase=True,
                byte_unigrams=True,
                )
            ),
            ('dict', DictVectorizer()),   # This will convert python dicts into efficient sparse data structures
            ('tfidf', TfidfTransformer()),
            ('clf', LogisticRegression(max_iter=500, verbose=2, C=100., solver='sag')),
        ]
    )

    # create dataloaders
    train_data, dev_data = read_corpus('train', 0.75)
    test_data, _ = read_corpus('test')

    text_clf.fit(train_data[:, 0], train_data[:, 1])  # this make take a moment with large corpora and/or large feature sets

    with open("unigram_dict_vectorizer.pkl", 'wb') as file:
        pickle.dump(text_clf['dict'], file)

    with open("unigram_tfidf.pkl", 'wb') as file:
        pickle.dump(text_clf['tfidf'], file)

    with open("unigram_logreg.pkl", 'wb') as file:
        pickle.dump(text_clf['clf'], file)

    print(classification_report(dev_data[:,1], text_clf.predict(dev_data[:, 0])))


if __name__ == "__main__":
    main()
