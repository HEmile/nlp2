from collections import Counter
import math
import numpy as np
import aer


def convert_to_ids(data):
    ids = {}
    _id = 0
    new_sentences = []
    for set in data:
        converted = []
        for s in set:
            sent = []
            for word in s:
                if word in ids:
                    sent.append(ids[word])
                else:
                    ids[word] = _id
                    _id += 1
                    sent.append(_id)
            if sent:
                converted.append(sent)
        new_sentences.append(converted)
    return new_sentences, _id + 1


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
            sent.append('NULL')
    french, F_vocab_size = convert_to_ids([french, frenchVal])
    return english, french, F_vocab_size, E_vocab_size


def init_t(english, french, E_vocab_size, F_vocab_size):
    combs = set()
    for k in range(len(english)):
        fdata = french[k]
        edata = english[k]
        for e in edata:
            if e != 'NULL':
                for f in fdata:
                    combs.add((f, e))
    print('Computing total counts per English words')
    count_tots = Counter()
    for _, e in combs:
        count_tots[e] += 1
    print('Computing initial chances')
    chances = np.empty(E_vocab_size)
    for e in count_tots.keys():
        chances[e] = 1 / count_tots[e]
    del count_tots
    print('Assigning t dictionary')
    t = {}
    for f, e in combs:
        t[f, e] = chances[e]
    print('Initialising NULL words')
    for f in range(F_vocab_size):
        t[f, 'NULL'] = 1 / F_vocab_size
    return t


# Runs one iteration of the EM algorithm and
# returns the new t matrix
def em_iteration(english, french, t):
    print('expectation')
    align_pairs = Counter()
    tot_align = Counter()
    for k in range(len(english)):
        fdata = french[k]
        edata = english[k]
        for f in fdata:
            norm = 0
            for e in edata:
                norm += t[f, e]
            for e in edata:
                delta = t[f, e] / norm
                align_pairs[e, f] += delta
                tot_align[e] += delta
    print('maximization')
    for e, f in align_pairs.keys():
        t[f, e] = align_pairs[e, f] / tot_align[e]
    return t

def main():
    english, french, F_vocab_size, E_vocab_size = init_data()
    english, french = english[0], french[0]
    print(E_vocab_size)
    print(F_vocab_size)

    # Init t uniformly
    # t = np.full((F_vocab_size, E_vocab_size + 1), 1/F_vocab_size)

    t = init_t(english, french, F_vocab_size, E_vocab_size)

    diff = 5
    prev = 1000

    # Train using EM
    while diff > 1:
        ent = entropy(english, french, t)
        print(ent)
        t = em_iteration(english, french, t)
        diff = prev - ent
        prev = ent


# Compute AER per iteration over validation data
def aer_metric():
    english, french, F_vocab_size, E_vocab_size = init_data()
    train_english, train_french = english[0], french[0]

    # Init t uniformly
    t = init_t(train_english, train_french, F_vocab_size, E_vocab_size)

    gold_sets = aer.read_naacl_alignments('validation/dev.wa.nonullalign')

    for s in range(4):
        t = em_iteration(train_english, train_french, t)

        # TODO: from t to predictions on validation

        metric = aer.AERSufficientStatistics()
        # then we iterate over the corpus 
        for gold, pred in zip(gold_sets, predictions):
            metric.update(sure=gold[0], probable=gold[1], predicted=pred)
        # AER
        print(metric.aer())


if __name__ == '__main__':
    main()
