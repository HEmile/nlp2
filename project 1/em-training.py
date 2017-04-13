from collections import Counter
import itertools
import math
import numpy as np


def convert_to_ids(sentences):
    ids = {}
    id = 0
    new_sentences = []
    for s in sentences:
        sent = []
        if len(new_sentences) > 5000:
            break
        for word in s:
            if word in ids:
                sent.append(ids[word])
            else:
                ids[word] = id
                id += 1
                sent.append(id)
        new_sentences.append(sent)
    return new_sentences, id + 1


def pos_alignments(l, m):
    return itertools.product(range(l), repeat=m)


# Using log likelihood as described here
# https://courses.engr.illinois.edu/cs498jh/HW/HW4.pdf
def likelihood(english, french, t):
    sum = 0
    for k in range(len(english)):
        edata = english[k]
        fdata = french[k]
        sent_sum = -len(fdata)*math.log(len(edata) + 1)
        for f in fdata:
            sum = 0
            for e in edata:
                sum += t[f, e]
            sent_sum += math.log(sum)
        sum += sent_sum
    return sum

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
    return (english, french, F_vocab_size, E_vocab_size)


def main():
    english, french, F_vocab_size, E_vocab_size = init_data()
    print(E_vocab_size)
    print(F_vocab_size)

    # Init t uniformly
    t = np.full((F_vocab_size, E_vocab_size + 1), 1/F_vocab_size)

    # Train using EM
    for s in range(4):
        print(s)
        print(likelihood(english, french, t))
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
        for f in range(F_vocab_size):
            for e in range(E_vocab_size):
                t[f, e] = align_pairs[e, f] / tot_align[e]


if __name__ == '__main__':
    main()
