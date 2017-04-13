from collections import Counter
import itertools
import math
import numpy as np


def convert_to_ids(sentences):
    ids = {}
    _id = 0
    new_sentences = []
    for s in sentences:
        sent = []
        for word in s:
            if word in ids:
                sent.append(ids[word])
            else:
                ids[word] = _id
                _id += 1
                sent.append(_id)
        new_sentences.append(sent)
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

# Set up the data
def init_data():
    english = []
    french = []
    with open('training/hansards.36.2.e', encoding='utf8') as f:
        for l in f:
            sent = l.split()
            english.append(sent)

    with open('training/hansards.36.2.f', encoding='utf8') as f:
        for l in f:
            sent = l.split()
            french.append(sent)

    english, E_vocab_size = convert_to_ids(english)
    french, F_vocab_size = convert_to_ids(french)
    return english, french, F_vocab_size, E_vocab_size


def main():
    english, french, F_vocab_size, E_vocab_size = init_data()
    print(E_vocab_size)
    print(F_vocab_size)

    # Init t uniformly
    # t = np.full((F_vocab_size, E_vocab_size + 1), 1/F_vocab_size)

    t = {}
    for k in range(len(english)):
        fdata = french[k]
        edata = english[k]
        for e in edata:
            for f in fdata:
                t[f, e] = 1
    print('Computing initial chances')
    for e in range(E_vocab_size):
        chance = 1 / sum([1 if (f, e) in t else 0 for f in range(F_vocab_size)])
        for f in range(F_vocab_size):
            if (f, e) in t:
                t[f, e] = chance

    diff = 5
    prev = 1000
    
    # Train using EM
    while diff > 1:
        ent = entropy(english, french, t)
        print(ent)
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
        for f in range(F_vocab_size):
            for e in range(E_vocab_size):
                if (f, e) in t:
                    t[f, e] = align_pairs[e, f] / tot_align[e]
        diff = prev - ent
        prev = ent


if __name__ == '__main__':
    main()
