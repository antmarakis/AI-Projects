import heapq
import bisect
import random
import string
import pandas as pd
from collections import Counter, defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import brown

# table = str.maketrans({key: None for key in string.punctuation})
# common_words = [w for w, _ in Counter(brown.words()).most_common(75)]


def predictions(test_x, nBS):
    d = []
    indices = ['"id"', '"EAP"', '"HPL"', '"MWS"']
    for index, row in test_x.iterrows():
        print(index)
        i, t = row['id'], row['text']
        p = recognize(t, nBS)
        d.append({'id': i, 'EAP': p['EAP'], 'HPL': p['HPL'], 'MWS': p['MWS']})
    
    return pd.DataFrame(data=d)


def NaiveBayes(dist):
    """A simple naive bayes classifier that takes as input a dictionary of
    CountingProbDist objects and classifies items according to these distributions.
    The input dictionary is in the following form:
        ClassName: Counter"""
    attr_dist = {c_name: count_prob for c_name, count_prob in dist.items()}

    def predict(example):
        """Predict the target value for example. Calculate probabilities for each
        class and pick the max."""
        def class_prob(target):
            attr = attr_dist[target]
            return product([attr[a] for a in example])

        pred = {t: class_prob(t) for t in dist.keys()}
        total = sum(pred.values())

        if total == 0:
            pred = predict(example[:int(2*len(example)/3)])
        else:
            for k, v in pred.items():
                pred[k] = v / total
        return pred

    return predict


def weighted_sampler(seq, weights):
    """Return a random-sample function that picks from seq weighted by weights."""
    totals = []
    for w in weights:
        totals.append(w + totals[-1] if totals else w)

    return lambda: seq[bisect.bisect(totals, random.uniform(0, totals[-1]))]


def product(numbers):
    """Return the product of the numbers, e.g. product([2, 3, 10]) == 60"""
    result = 1
    for x in numbers:
        result *= x
    return result


def remove_stopwords(sentence):
    words = word_tokenize(sentence.translate(table))
    return ' '.join([w for w in words if w not in stopwords.words('english')])


def remove_most_common(sentence):
    words = word_tokenize(sentence)
    return ' '.join([w for w in words if w not in common_words])


def recognize(sentence, nBS):
    sentence = sentence.lower()
    sentence_words = word_tokenize(sentence)
    
    return nBS(sentence_words)


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
            EAP += " " + t
        elif a == 'HPL':
            HPL += " " + t
        elif a == 'MWS':
            MWS += " " + t

    write_to_file('eap.txt', EAP)
    write_to_file('hpl.txt', HPL)
    write_to_file('mws.txt', MWS)


def write_to_file(fname, text):
    f = open(fname, 'w', encoding='utf-8')
    f.write(text)
    f.close()


def read_from_file(fnames):
    texts = []
    for fname in fnames:
        f = open(fname, 'r', encoding='utf-8')
        texts.append(word_tokenize(f.read()))
        f.close()
    return texts