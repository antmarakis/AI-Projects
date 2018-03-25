import heapq
import bisect
import random
import string
import pandas as pd
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import brown
import decimal
from decimal import Decimal

decimal.getcontext().prec = 100

# table = str.maketrans({key: None for key in string.punctuation})
# common_words = [w for w, _ in Counter(brown.words()).most_common(75)]


def predictions(test_x, nBS):
    d = []
    for index, row in test_x.iterrows():
        print(index)
        i, t = row['id'], row['text']
        p = recognize(t, nBS)
        d.append({'id': i, 'EAP': p['EAP'], 'HPL': p['HPL'], 'MWS': p['MWS']})
    
    return pd.DataFrame(data=d)


def recognize(sentence, nBS):
    return nBS(word_tokenize(sentence.lower()))


def precise_product(numbers):
    result = 1
    for x in numbers:
        result *= Decimal(x)
    return result


def NaiveBayes(dist):
    """A simple naive bayes classifier that takes as input a dictionary of
    Counter distributions and can then be used to find the probability
    of a given item belonging to each class.
    The input dictionary is in the following form:
        ClassName: Counter"""
    attr_dist = {c_name: count_prob for c_name, count_prob in dist.items()}

    def predict(example):
        """Predict the probabilities for each class."""
        def class_prob(target, e):
            attr = attr_dist[target]
            return precise_product([attr[a] for a in e])

        pred = {t: class_prob(t, example) for t in dist.keys()}

        total = sum(pred.values())
        for k, v in pred.items():
            pred[k] = v / total

        return pred

    return predict


def remove_stopwords(sentence):
    # Not used
    words = word_tokenize(sentence)
    return ' '.join([w for w in words if w not in stopwords.words('english')])


def remove_most_common(sentence):
    # Not used
    words = word_tokenize(sentence)
    return ' '.join([w for w in words if w not in common_words])


def create_dist(text):
    c = Counter(text)

    least_common = c.most_common()[-1][1]
    total = sum(c.values())
    
    for k, v in c.items():
        c[k] = v/total

    return defaultdict(lambda: min(c.values()), c)


def parse_text(X):
    EAP, HPL, MWS = "", "", ""

    for i, row in X.iterrows():
        a, t = row['author'], row['text']
        print(i)
        
        if a == 'EAP':
            EAP += " " + t.lower()
        elif a == 'HPL':
            HPL += " " + t.lower()
        elif a == 'MWS':
            MWS += " " + t.lower()

    return word_tokenize(EAP), word_tokenize(HPL), word_tokenize(MWS)
