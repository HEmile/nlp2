from collections import Counter
import itertools
import math
import numpy as np
english = []
french = []
with open('training/hansards.36.2.e', 'r', encoding='utf8') as f:
    for l in f:
        sent = l.split()
        english.append(sent)

with open('training/hansards.36.2.f', 'r', encoding='utf8') as f:
    for l in f:
        sent = l.split()
        french.append(sent)

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

english, E_vocab_size = convert_to_ids(english)
french, F_vocab_size = convert_to_ids(french)

print(E_vocab_size)
print(F_vocab_size)

def pos_alignments(l, m):
    return itertools.product(range(l), repeat=m)

def likelihood(english, french, t):
    sum = 0
    for k in range(len(english)):
        print(k)
        sumlog = 0
        edata = english[k]
        fdata = french[k]
        for alignment in pos_alignments(len(edata), len(fdata)):
            prod = 1
            for i in range(len(fdata)):
                prod *= t[fdata[i], edata[alignment[i]]]
            sumlog += prod
        sum += math.log(sumlog)
    return sum

t = np.full((F_vocab_size, E_vocab_size + 1), 1/F_vocab_size)

for s in range(4):
    print(s)
    print(likelihood(english, french, t))
    align_pairs = Counter()
    tot_align = Counter()
    for k in range(len(english)):
        fdata = french[k]
        edata = english[k]
        for i in range(len(fdata)):
            f = fdata[i]
            norm = 0
            for j in range(len(edata)):
                norm += t[f, edata[j]]
            for j in range(len(edata)):
                e = edata[j]
                delta = t[f, e] / norm
                align_pairs[e, f] += delta
                tot_align[e] += delta
    for f in range(F_vocab_size):
        for e in range(E_vocab_size):
            t[f, e] = align_pairs[e, f] / tot_align[e]