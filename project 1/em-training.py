from collections import Counter
import math
import numpy as np
import aer
import operator
from scipy.special import digamma, loggamma, gammaln, gamma
import pickle
import matplotlib.pyplot as plt


JUMP_LENGTH = 100

IBM_1_ITERATIONS = 10
IBM_2_ITERATIONS = 5

LOAD_PATH = 'IBM1-em10.t'

VOCABULARY = 15000

def convert_to_ids(data):
    counts = Counter()
    for set1 in data:
        for s in set1:
            for word in s:
                counts[word] += 1
    st = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
    ids = {}
    for i in range(VOCABULARY):
        ids[st[i][0]] = i + 1
    for i in range(VOCABULARY, len(st)):
        ids[st[i][0]] = VOCABULARY + 1
    new_sentences = []
    for set1 in data:
        converted = []
        for s in set1:
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
                value = t[french[i], english[j]] #* q[jump(j, i, len(english), len(french))]
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
    englishTest = read_dataset('testing/test/test.e')
    frenchTest = read_dataset('testing/test/test.f')

    english, E_vocab_size = convert_to_ids([english, englishVal, englishTest])
    for set in english:
        for sent in set:
            sent.append(0) # Append NULL WORD
    french, F_vocab_size = convert_to_ids([french, frenchVal, frenchTest])
    long_s = []
    for k in range(len(french[0])):
        if len(french[0][k]) >= JUMP_LENGTH or len(english[0][k]) >= JUMP_LENGTH:
            long_s.append(k)
    english[0] = np.delete(english[0], long_s)
    french[0] = np.delete(french[0], long_s)
    return english, french, E_vocab_size, F_vocab_size


def init_t_uniform(english, french, E_vocab_size, F_vocab_size):
    print('Initialising t array uniform')
    t = np.full((F_vocab_size, E_vocab_size), 1/(F_vocab_size - 1))
    return t

def init_t_random(english, french, E_vocab_size, F_vocab_size):
    print('Initialising t array random')
    randoms = np.random.rand(F_vocab_size, E_vocab_size)
    t = np.divide(randoms, sum(randoms))
    return t


def init_t_ibm1_em(english, french, E_vocab_size, F_vocab_size, val_english, val_french):
    print('Initialising t array using precomputed IBM 1 model')
    t = init_t_uniform(english, french, E_vocab_size, F_vocab_size)
    for i in range(IBM_1_ITERATIONS):
        #t, _, _ = em_iteration_ibm1(english, french, t, None, E_vocab_size, F_vocab_size)
        t = vb_iteration(english, french, t, None, E_vocab_size, F_vocab_size)
        me = aer_metric(val_english, val_french, t, _)
    with open('IBM1-em' + str(IBM_1_ITERATIONS) + '.t', 'wb') as f:
        pickle.dump(t, f)
    return t


def init_t_pkl(english, french, E_vocab_size, F_vocab_size):
    with open(LOAD_PATH, 'rb') as f:
        return pickle.load(f)


def init_q():
    q = {}
    for i in range(-JUMP_LENGTH, JUMP_LENGTH):
        q[i] = 1/(2*JUMP_LENGTH + 1)
    q['NULL'] = 1/(2*JUMP_LENGTH + 1)
    return q


def jump(i, j, m, n):
    return i - math.floor(j*m/n) if i != 0 else 'NULL'


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


def vb_iteration(english, french, t, q, E_vocab_size, F_vocab_size, alpha=0.0001):
    print('expectation')
    align_pairs = Counter()
    entropy = 0
    
    #Fout volgens mij: 
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
        
    #elb = elbo(english, french, align_pairs, t, F_vocab_size, E_vocab_size, alpha )
    #print('ELBO:', elb)
    return t

def elbo(english, french, align, t, F_vocab_size, E_vocab_size, alpha):
    print("Computing ELBO")
    mle = 0
    for k in range(len(english)):
        fdata = french[k]
        edata = english[k]
        sum_n = 0
        for f in fdata:
            sum_m = 0
            for e in edata:
                sum_m += t[f,e]
            sum_n += math.log(sum_m)
        mle += sum_n
    
    print("Part 2 -ELBO")
    #exp_log = np.zeros((F_vocab_size, E_vocab_size))
    #for (f,e), value in np.ndenumerate(t):
    #     exp_log[f,e] = ((digamma(value) - digamma(sum(t[:,e]) - value)) * (alpha - value) + loggamma(value) - loggamma(alpha))
    #     print(f,e)
    
    #print("Part 2.1 - ELBO")
    #kl= 0     
    #for e in range(E_vocab_size):
    #    kl_e = sum(exp_log[:,e]) + loggamma(alpha*F_vocab_size) - loggamma(sum(t[:,e]))
    #    kl += kl_e
    
    kl = 0        
    for e in range(E_vocab_size):
        for f in range(F_vocab_size): 
           first_p = (digamma(t[f,e]) - digamma(sum(t[:,e]) - t[f,e])) * (alpha - t[f,e]) + loggamma(t[f,e]) - loggamma(alpha)
           second_p = loggamma(alpha*F_vocab_size) - loggamma(sum(t[:,e]))
           kl_e = first_p.real + second_p.real
        kl += kl_e
    print(kl)
    elb = -kl + mle
    return elb

def plots():
    iterations = [1,2,3,4,5,6,7,8,9,10]
    
    #Na een run overgenomen (alpha = 0.001 en vocab = 15000):
    IBM1_ent = [127.42, 36.56, 23.98, 20.15, 18.85, 18.26, 17.95, 17.76, 17.65, 17.57]
    IBM1_aer = [0.641, 0.625, 0.622, 0.624, 0.627, 0.628, 0.626, 0.625, 0.626, 0.625]
    vb_IBM1_ent = [127.42, 44.66, 28.605, 25.579, 24.509, 24.007, 23.732, 23.565, 23.457, 23.384]
    vb_IBM1_aer = [0.655, 0.638, 0.632, 0.628, 0.629, 0.633, 0.632, 0.632, 0.629, 0.628]
    vb_IBM1_elbo = []
    
    #alpha = 0.01
    vb2_IBM1_ent = [127.42, 47.46, 31.62, 28.51, 27.49, 27.02, 26.77, 26.62, 26.53, 26.47]
    vb2_IBM1_aer = [0.64, 0.66, 0.63, 0.632, 0.63, 0.631, 0.628, 0.627, 0.627, 0.626]
    
    #alpha = 0.0001
    vb3_IBM1_ent = [127.42, 44.05, 27.91, 24.87, 23.77, 23.25, 22.97, 22.80, 22.69, 22.61]
    vb3_IBM1_aer = [0.65, 0.637, 0.631, 0.630, 0.630, 0.634, 0.633, 0.633, 0.634, 0.633]
    
    # alpha = 0.001 en voacb = 15000
    uni_IBM2_ent = [232.12, 107.83, 84.51, 76.13, 73.68]
    uni_IBM2_aer = [0.547, 0.539, 0.516, 0.526, 0.524]
    rand_IBM2_ent1 = [232.36, 108.50, 85.63, 76.91, 74.15]
    rand_IBM2_aer1 = [0.596, 0.542, 0.526, 0.516, 0.521]
    rand_IBM2_ent2 = [232.46, 108.67, 85.69, 76.99, 74.31]
    rand_IBM2_aer2 = [0.608, 0.567, 0.548, 0.543, 0.536]
    rand_IBM2_ent3 = [232.27, 108.49, 85.38, 76.65, 74.04]
    rand_IBM2_aer3 = [0.61, 0.562, 0.530, 0.528, 0.531]
    ibm_IBM2_ent = [122.22, 79.57, 75.74, 74.58, 73.96]
    ibm_IBM2_aer = [0.542, 0.531, 0.530, 0.530, 0.531 ]
    
    plt.plot(iterations, IBM1_aer, '--o', label='MLE')
    plt.plot(iterations, vb_IBM1_aer, 'r--o', label='VB')
    plt.xlabel('Iterations')
    plt.ylabel('Entropy')
    plt.legend()
    plt.show()


def test(t, q, name, english, french):
    
    #AER for test + writing to file
    gold_sets = aer.read_naacl_alignments('testing/answers/test.wa.nonullalign')
    predictions = []

    with open(name, 'w') as f:
        
        for k in range(len(french)):
            english_sen = english[k]
            french_sen = french[k]
            sen_best = set()
            for i in range(len(french_sen)):
                old_val = 0
                for j in range(len(english_sen)):
                    value = t[french_sen[i], english_sen[j]] #* q[jump(j, i, len(english), len(french))]
                    if value >= old_val:
                        best = (j, i)
                        old_val = value
                if english[best[0]] != 0:
                    sen_best.add(best)
                    f.write(str(k+1) + " " + str(best[0]) + " " + str(best[1]) +'\n')
            predictions.append(sen_best)

    metric = aer.AERSufficientStatistics()
    # then we iterate over the corpus
    for gold, pred in zip(gold_sets, predictions):
        metric.update(sure=gold[0], probable=gold[1], predicted=pred)
    # AER
    me = metric.aer()
    print("AER:", me)
    

def main():
    english, french, E_vocab_size, F_vocab_size = init_data()
    english_test, french_test = english[2], french[2]
    english_val, french_val = english[1], french[1]
    english, french = english[0], french[0]

    print(E_vocab_size)
    print(F_vocab_size)

    # Init t uniformly
    #t = np.full((F_vocab_size, E_vocab_size + 1), 1/F_vocab_size)

    #Init t random
    #t = init_t_random(english, french, E_vocab_size, F_vocab_size)

    #t = init_t_ibm1_em(english, french, E_vocab_size, F_vocab_size, english_val, french_val)
    q = init_q()
    
    t = init_t_pkl(english, french, E_vocab_size, F_vocab_size)
    
    test(t, _, 'ibm1.vb.naacl', english_test, french_test)

    print('STARTING IBM MODEL 2')
    # Train using EM
    for i in range(IBM_2_ITERATIONS):
        t, q, ent = em_iteration_ibm2(english, french, t, q, E_vocab_size, F_vocab_size)
        #aer_metric(english_val, french_val, t, q)
        
    #test(t, q, 'ibm2.mle.naacl', english_test, french_test)

if __name__ == '__main__':
    main()
    #plots()


