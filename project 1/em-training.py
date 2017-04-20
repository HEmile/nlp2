from collections import Counter
import math
import numpy as np
import aer
import operator
from scipy.special import digamma


def convert_to_ids(data, truncate_size=10000):
    counts = Counter()
    for set in data:
        for s in set:
            for word in s:
                counts[word] += 1
    st = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
    ids = {}
    for i in range(truncate_size):
        ids[st[i][0]] = i + 1
    for i in range(truncate_size, len(st)):
        ids[st[i][0]] = truncate_size + 1
    new_sentences = []
    for set in data:
        converted = []
        for s in set:
            sent = []
            for word in s:
                sent.append(ids[word])
            if sent:
                converted.append(sent)
        new_sentences.append(converted)
    return new_sentences, truncate_size + 2 # !???


# Using log entropy as described here
# https://courses.engr.illinois.edu/cs498jh/HW/HW4.pdf
def entropy(english, french, t):
    _sum = 0
    for k in range(len(english)):
        edata = english[k]
        fdata = french[k]
        sum_n = 0
        for f in fdata:
            sum_m = 0
            for e in edata:
                sum_m += t[f, e]
            sum_n += math.log(sum_m)
        _sum += sum_n
    return _sum * -1 / len(english)


def read_dataset(path):
    sentences = []
    with open(path, encoding='utf8') as f:
        for l in f:
            sent = l.split()
            sentences.append(sent)
    return sentences


# Set up the data
def init_data():
    english = read_dataset('training/hansards.36.2.e')
    french = read_dataset('training/hansards.36.2.f')
    englishVal = read_dataset('validation/dev.e')
    frenchVal = read_dataset('validation/dev.f')

    english, E_vocab_size = convert_to_ids([english, englishVal])
    for set in english:
        for sent in set:
            sent.append(0) # Append NULL WORD
    french, F_vocab_size = convert_to_ids([french, frenchVal])
    return english, french, E_vocab_size, F_vocab_size


def init_t(english, french, E_vocab_size, F_vocab_size):
    print('Initialising t array')
    t = np.full((F_vocab_size, E_vocab_size), 1/(F_vocab_size - 1)) # You also initialize f = 0, even though it's never used
    return t

# Runs one iteration of the EM algorithm and
# returns the new t matrix
def em_iteration(english, french, t, E_vocab_size, F_vocab_size):
    print('expectation')
    align_pairs = Counter()
    tot_align = Counter()
    for k in range(len(english)):
        fdata = french[k]
        edata = english[k]
        for f in fdata:
            norm = sum([t[f, e] for e in edata])
            for e in edata:
                delta = t[f, e] / norm
                align_pairs[e, f] += delta
                tot_align[e] += delta
    print('maximization')
    for e, f in align_pairs.keys():
        t[f, e] = align_pairs[e, f] / tot_align[e]
    return t


def vb_iteration(english, french, t, E_vocab_size, F_vocab_size, alpha=0.001):
    print('expectation')
    align_pairs = Counter()
    for k in range(len(english)):
        fdata = french[k]
        edata = english[k]
        for f in fdata:
            norm = sum([t[f, e] for e in edata])
            for e in edata:
                delta = t[f, e] / norm
                align_pairs[e, f] += delta
    print('Maximization')
    sum_psis = np.empty(E_vocab_size)
    for e in range(E_vocab_size):
        sum_l = sum([align_pairs[e, f] for f in range(F_vocab_size)])
        sum_l += alpha * F_vocab_size
        sum_psis[e] = digamma(sum_l)
    for e, f in align_pairs.keys():
        t[f, e] = math.exp(digamma(align_pairs[e, f] + alpha) - sum_psis[e])
    return t

def main():
    english, french, E_vocab_size, F_vocab_size = init_data()
    english, french = english[0], french[0]
    print(E_vocab_size)
    print(F_vocab_size)

    # Init t uniformly
    # t = np.full((F_vocab_size, E_vocab_size + 1), 1/F_vocab_size)

    t = init_t(english, french, E_vocab_size, F_vocab_size)

    diff = 5
    prev = 1000

    # Train using EM
    while diff > 1:
        ent = entropy(english, french, t)
        print(ent)
        t = vb_iteration(english, french, t, E_vocab_size, F_vocab_size)
        diff = prev - ent
        prev = ent


# Compute AER per iteration over validation data
def aer_metric():
    english, french, E_vocab_size, F_vocab_size = init_data()
    train_english, train_french = english[0], french[0]

    val_english, val_french = english[1], french[1]

    # Init t uniformly
    t = init_t(train_english, train_french, E_vocab_size, F_vocab_size)

    gold_sets = aer.read_naacl_alignments('validation/dev.wa.nonullalign')
    metrics = []

    for s in range(4):
<<<<<<< HEAD
        t = em_iteration(train_english, train_french, t)
        
=======
        t = em_iteration(train_english, train_french, t, E_vocab_size, F_vocab_size)

>>>>>>> origin/master
        predictions = []
        for k in range(len(val_french)):
            english = val_english[k]
            french = val_french[k]
            sen = set()
            for i in range(len(french)):
                old_val = 0
                for j in range(len(english)):
                    value = t[french[i], english[j]]
                    if value >= old_val:
                        best = (j, i)
                        old_val = value
                if english[best[0]] != 0:
                    sen.add(best)
            predictions.append(sen)

        metric = aer.AERSufficientStatistics()
        # then we iterate over the corpus
        for gold, pred in zip(gold_sets, predictions):
            metric.update(sure=gold[0], probable=gold[1], predicted=pred)
        # AER
        me = metric.aer()
        print(me)
        
        metrics.append(me)


if __name__ == '__main__':
<<<<<<< HEAD
    # main()
    aer_metric() #On validation data
=======
    main()
    # aer_metric()
>>>>>>> origin/master

