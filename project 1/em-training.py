from collections import Counter
import math
import numpy as np
import aer
import operator
from scipy.special import digamma
import pickle
import matplotlib.pyplot as plt


JUMP_LENGTH = 100

IBM_1_ITERATIONS = 10
IBM_2_ITERATIONS = 5

LOAD_PATH = 'IBM1-em-10.t'

VOCABULARY = 15000

def convert_to_ids(data):
    counts = Counter()
    for set in data:
        for s in set:
            for word in s:
                counts[word] += 1
    st = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
    ids = {}
    for i in range(VOCABULARY):
        ids[st[i][0]] = i + 1
    for i in range(VOCABULARY, len(st)):
        ids[st[i][0]] = VOCABULARY + 1
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
    return new_sentences, VOCABULARY + 2 # !???


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


def aer_metric(val_english, val_french, t, q):
    gold_sets = aer.read_naacl_alignments('validation/dev.wa.nonullalign')
    predictions = []
    for k in range(len(val_french)):
        english = val_english[k]
        french = val_french[k]
        sen = set()
        for i in range(len(french)):
            old_val = 0
            for j in range(len(english)):
                value = t[french[i], english[j]] * q[jump(j, i, len(english), len(french))]
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
    print("AER:", me)
    return me

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
    long_s = []
    for k in range(len(french[0])):
        if len(french[0][k]) >= JUMP_LENGTH or len(english[0][k]) >= JUMP_LENGTH:
            long_s.append(k)
    english[0] = np.delete(english[0], long_s)
    french[0] = np.delete(french[0], long_s)
    return english, french, E_vocab_size, F_vocab_size


def init_t_uniform(english, french, E_vocab_size, F_vocab_size):
    print('Initialising t array')
    t = np.full((F_vocab_size, E_vocab_size), 1/(F_vocab_size - 1))
    return t


def init_t_ibm1_em(english, french, E_vocab_size, F_vocab_size, val_english, val_french):
    print('Initialising t array using precomputed IBM 1 model')
    t = init_t_uniform(english, french, E_vocab_size, F_vocab_size)
    for i in range(IBM_1_ITERATIONS):
        t, _, _ = em_iteration_ibm1(english, french, t, None, E_vocab_size, F_vocab_size)
        #t = vb_iteration(english, french, t, None, E_vocab_size, F_vocab_size)
        #me = aer_metric(val_english, val_french, t, _)
    with open('IBM1-em' + str(IBM_1_ITERATIONS) + '.t', 'wb') as f:
        pickle.dump(t, f)
    return t


def init_t_pkl(english, french, E_vocab_size, F_vocab_size):
    with open(LOAD_PATH, 'rb') as f:
        return pickle.load(f)

def init_q():
    q = {}
    for i in range(-JUMP_LENGTH, JUMP_LENGTH):
        q[i] = 1/(2*JUMP_LENGTH)
    return q


def jump(i, j, m, n):
    return min(max(i - math.floor(j*m/n), -JUMP_LENGTH), JUMP_LENGTH - 1)

# Runs one iteration of the EM algorithm and
# returns the new t matrix
def em_iteration_ibm1(english, french, t, q, E_vocab_size, F_vocab_size):
    print('expectation')
    align_pairs = Counter()
    tot_align = Counter()
    entropy = 0
    for k in range(len(english)):
        fdata = french[k]
        edata = english[k]
        sum_n = 0
        for f in fdata:
            norm = 0
            for e in edata:
                norm += t[f, e]
            sum_n += math.log(norm)
            for e in edata:
                delta = t[f, e] / norm
                align_pairs[e, f] += delta
                tot_align[e] += delta
        entropy += sum_n

    entropy *= -1 / len(english)
    print('Entropy:', entropy)
    print('maximization')
    for e, f in align_pairs.keys():
        t[f, e] = align_pairs[e, f] / tot_align[e]
    return t, q, entropy


def em_iteration_ibm2(english, french, t, q, E_vocab_size, F_vocab_size):
    print('expectation')
    align_pairs = Counter()
    tot_align = Counter()
    jump_cs = Counter()
    entropy = 0
    for k in range(len(english)):
        fdata = french[k]
        edata = english[k]
        sum_n = 0
        for j in range(len(fdata)):
            f = fdata[j]
            norm = 0
            for i in range(len(edata)):
                e = edata[i]
                norm += q[jump(i, j, len(edata), len(fdata))] * t[f, e]
            sum_n += math.log(norm)
            for i in range(len(edata)):
                e = edata[i]
                jump_i = jump(i, j, len(edata), len(fdata))
                delta = q[jump_i] * t[f, e] / norm
                align_pairs[e, f] += delta
                tot_align[e] += delta
                jump_cs[jump_i] += delta
        entropy += sum_n

    entropy *= -1 / len(english)
    print('Entropy:', entropy)
    print('maximization')
    for e, f in align_pairs.keys():
        t[f, e] = align_pairs[e, f] / tot_align[e]
    tot_jump = sum(jump_cs[i] for i in range(-JUMP_LENGTH, JUMP_LENGTH))
    for i in range(-JUMP_LENGTH, JUMP_LENGTH):
        q[i] = jump_cs[i] / tot_jump
    return t, q, entropy


def vb_iteration(english, french, t, q, E_vocab_size, F_vocab_size, alpha=0.001):
    print('expectation')
    align_pairs = Counter()
    entropy = 0
    for k in range(len(english)):
        fdata = french[k]
        edata = english[k]
        sum_n = 0
        for f in fdata:
            norm = 0
            for e in edata:
                norm += t[f, e]
            sum_n += math.log(norm)
            for e in edata:
                delta = t[f, e] / norm
                align_pairs[e, f] += delta
        entropy += sum_n

    entropy *= -1 / len(english)
    print('Entropy:', entropy)
    print('Maximization')
    sum_psis = np.empty(E_vocab_size)
    for e in range(E_vocab_size):
        sum_l = sum([align_pairs[e, f] for f in range(F_vocab_size)])
        sum_l += alpha * F_vocab_size
        sum_psis[e] = digamma(sum_l)
    for e, f in align_pairs.keys():
        t[f, e] = math.exp(digamma(align_pairs[e, f] + alpha) - sum_psis[e])
    return t

def plots():
    iterations = [1,2,3,4,5,6,7,8,9,10]
    IBM1_ent = [127.42, 36.56, 23.98, 20.15, 18.85, 18.26, 17.95, 17.76, 17.65, 17.57]
    IBM1_aer = [0.641, 0.625, 0.622, 0.624, 0.627, 0.628, 0.626, 0.625, 0.626, 0.625]
    vb_IBM1_ent = [127.42, 44.66, 28.605, 25.579, 24.509, 24.007, 23.732, 23.565, 23.457, 23.384]
    vb_IBM1_aer = [0.655, 0.638, 0.632, 0.628, 0.629, 0.633, 0.632, 0.632, 0.629, 0.628]
    plt.plot(iterations, IBM1_aer, '-o', iterations, vb_IBM1_aer, 'r-o')
    plt.xlabel('Iterations')
    plt.ylabel('AER')
    plt.show()

def main():
    english, french, E_vocab_size, F_vocab_size = init_data()
    english_val, french_val = english[1], french[1]
    english, french = english[0], french[0]
    print(E_vocab_size)
    print(F_vocab_size)

    # Init t uniformly
    # t = np.full((F_vocab_size, E_vocab_size + 1), 1/F_vocab_size)

    t = init_t_ibm1_em(english, french, E_vocab_size, F_vocab_size, english_val, french_val)
    q = init_q()

    diff = 5
    prev = 1000

    print('STARTING IBM MODEL 2')
    # Train using EM
    for i in range(IBM_2_ITERATIONS):
        t, q, ent = em_iteration_ibm2(english, french, t, q, E_vocab_size, F_vocab_size)
        print(aer_metric(english_val, french_val, t, q))
        diff = prev - ent
        prev = ent

if __name__ == '__main__':
    #main()
    plots()


