import sys
from helper import *
from earley import *
from libitg import *
from collections import defaultdict
from feature_helper import gradient, skip_bigrams, featurize_edges, likelihood, predict
import pickle
import os
import matplotlib.pyplot as plot
import numpy as np
import math
import random
import subprocess


LIMIT_TRANS_LENGTH = 3

PARTITION = 1

DATA_SET_INDEX = 0

SENTENCE_LENGTH = 10

BATCH_SIZE = 29

SGD_ITERATIONS = 10

LAMBDA_LR = 10

SIGMA = 10

GAMMA0 = 0.1

USE_SPARSE_F = True

USE_SKIP_DICT = True

USE_LOAD_W = False

LOAD_W_PATH = 'wsparse29-10-1000.pkl'

PRINT_COUNT = 2000  # Amount of training samples to do SGD over until we print val and test results


def prepare_val(skip_dict, weights_ibm):
    chi_val, eng_val = read_data('data/dev1.zh-en')
    pp = []
    for i in range(len(chi_val)):
        path = 'parses/val' + str(i) + '.pkl'
        if not os.path.exists(path):
            continue
        with open(path, 'rb') as f:
            (dx, dxy) = pickle.load(f)
        src_fsa = make_fsa(chi_val[i])
        fmapx = featurize_edges(dx, src_fsa, weights_ibm, skip_dict, sparse_del=USE_SPARSE_F, sparse_trans=USE_SPARSE_F,
                                sparse_ins=USE_SPARSE_F, use_skip_dict=USE_SKIP_DICT)
        fmapxy = featurize_edges(dxy, src_fsa, weights_ibm, skip_dict, sparse_del=USE_SPARSE_F, sparse_trans=USE_SPARSE_F,
                                 sparse_ins=USE_SPARSE_F, use_skip_dict=USE_SKIP_DICT, use_bispans=True)
        pp.append((dx, dxy, fmapx, fmapxy))
    return pp


def prepare_test(skip_dict, weights_ibm):
    chi_val, eng_val = read_data('data/dev2.zh-en')
    pp = []
    for i in range(len(chi_val)):
        path = 'parses/test' + str(i) + '.pkl'
        with open(path, 'rb') as f:
            (dx, _) = pickle.load(f)
        src_fsa = make_fsa(chi_val[i])
        fmapx = featurize_edges(dx, src_fsa, weights_ibm, skip_dict, sparse_del=USE_SPARSE_F, sparse_trans=USE_SPARSE_F,
                                sparse_ins=USE_SPARSE_F, use_skip_dict=USE_SKIP_DICT)
        pp.append((dx, fmapx))
    return pp


def main(parse=False, featurise=True, sgd=True, save_w=True, validate=True, test=True):
    chinese, english = read_data('data/training.zh-en')
    skip_dict = skip_bigrams(chinese)
    mn, mx = DATA_SET_INDEX * (len(chinese) // PARTITION), (DATA_SET_INDEX + 1) * (len(chinese) // PARTITION)
    chinese, english = chinese[mn: mx], english[mn: mx]
    lexicon, weights, ch_vocab, en_vocab, null_alligned = read_lexicon_ibm('lexicon')
    if validate:
        print('preparing validation set')
        val = prepare_val(skip_dict, weights)
    if test:
        print('preparing test set')
        tst = prepare_test(skip_dict, weights)

    if USE_LOAD_W:
        with open(LOAD_W_PATH, 'rb') as f:
            w = defaultdict(float, pickle.load(f))
    else:
        w = defaultdict(float)

    if not os.path.exists('parses'):
        os.makedirs('parses')

    if not os.path.exists('features'):
        os.makedirs('features')

    print('Parsing sentences', mn, 'to', mx)
    count = 0
    g_batch = defaultdict(float)
    count_batch = 0
    best_likelihood = -sys.maxsize
    t = 0
    for iter in range(SGD_ITERATIONS):
        print('STARTING SGD ITERATION', iter + 1)
        for i in range(len(chinese)):
            index = mn + i
            chi_src = chinese[i]
            en_src = english[i]
            chi_spl = chi_src.split()
            en_spl = en_src.split()

            if len(chi_spl) > SENTENCE_LENGTH or len(en_spl) > SENTENCE_LENGTH:
                continue

            path = "parses/" + str(index) + '.pkl'

            def map_unk(splt, vocab):
                for i in range(len(splt)):
                    if splt[i] not in vocab:
                        splt[i] = '-UNK-'
                return ' '.join(splt)
            chi_src = map_unk(chi_spl, ch_vocab)
            en_src = map_unk(en_spl, en_vocab)

            src_fsa = make_fsa(chi_src)
            tgt_fsa = make_fsa(en_src)

            if parse:
                if not os.path.exists(path):
                    print(index)
                    lexicon['-EPS-'] = set(null_alligned)
                    for c in chi_spl:  # Belangrijk voor report: Deze toevoegen zorgt ervoor dat heel veel parset
                        lexicon['-EPS-'] = lexicon['-EPS-'].union([lexicon[c][0]])

                    src_cfg = make_source_side_finite_itg(lexicon)
                    forest = earley(src_cfg, src_fsa, start_symbol=Nonterminal('S'), sprime_symbol=Nonterminal("D(x)"))
                    dx = make_target_side_finite_itg(forest, lexicon)
                    dxy = earley(dx, tgt_fsa, start_symbol=Nonterminal("D(x)"), sprime_symbol=Nonterminal('D(x, y)'))

                    if len(dxy) == 0:
                        continue

                    with open(path, 'wb') as f:
                        pickle.dump((dx, dxy), f)
            else:
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        dx, dxy = pickle.load(f)
                else:
                    continue

            count += 1
            count_batch += 1
            if sgd:
                dw, likel = gradient(dx, dxy, src_fsa, w, weights, skip_dict, index, featurise, SIGMA, USE_SPARSE_F, USE_SKIP_DICT)
                if dw:
                    for k, dwk in dw.items():
                        g_batch[k] += dwk
                if count_batch % BATCH_SIZE == 0:
                    gammat = GAMMA0 * (1 / (1 + GAMMA0 * LAMBDA_LR * t))
                    t += 1
                    for k, dwk in g_batch.items():
                        w[k] += gammat * dwk / BATCH_SIZE
                    g_batch = defaultdict(float)
            if count % PRINT_COUNT == 0:
                print(index)
                print(gammat)

                if validate:
                    lls = []
                    for vdx, vdxy, vfmapx, vfmapxy in val:
                        ll = likelihood(vdx, vdxy, None, w, None, None, sigma=SIGMA, fmapxn=vfmapx, fmapxy=vfmapxy)
                        lls.append(ll)
                    val_ll = sum(lls) / len(lls)

                    print(val_ll)
                if test:
                    predictions = []
                    for vdx, vfmapx in tst:
                        p = predict(vdx, vfmapx, w)
                        predictions.append(p)
                    if predict:
                        with open('predictions.txt', 'w') as f:
                            for p in predictions:
                                print(p, file=f)
                        # print(run(['perl',  'multi-bleu.perl', 'reference1.txt',  'predictions.txt']))
                        p = subprocess.Popen('perl multi-blue.perl reference1.txt < predictions.txt', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                        for line in p.stdout.readlines():
                            print(line)
                        p.wait()
                if save_w and (not validate or val_ll > best_likelihood):
                    modifier = 'sparse' if USE_SPARSE_F else ''
                    modifier += 'skip' if USE_SKIP_DICT else ''
                    modifier += str(BATCH_SIZE) + '-' + str(LAMBDA_LR) + '-' + str(SIGMA)
                    print('m:', modifier)
                    if save_w:
                        with open('w'+modifier+str(iter)+'.pkl', 'wb') as f:
                            pickle.dump(dict(w), f)
                            best_likelihood = val_ll
                count = 0


def test_gradient():
    ls_gr_x = []
    ls_gr_y = []
    ls_gr_ac = []
    ls_gr_cp = []
    ls_gr_ll = []
    sigma = 5
    chinese, english = read_data('data/training.zh-en')
    _, weights, _, _, _ = read_lexicon_ibm('lexicon')
    w1 = defaultdict(lambda: (random.random() - 0.5)*4)
    # w1 = defaultdict(float)
    for w in np.arange(-5, 5, 0.01):
        src_fsa = make_fsa(chinese[3])
        with open('parses/3.pkl', 'rb') as f:
            dx1, dxy1 = pickle.load(f)
        H = 0.0001
        key = 'type:target_length'
        w1[key] = w
        dw, likel1 = gradient(dx1, dxy1, src_fsa, w1, weights, None, 3, True, sigma)
        w2 = dict(w1)
        w2[key] = w + H
        _, likel2 = gradient(dx1, dxy1, src_fsa, w2, weights, None, 3, True, sigma)
        dwk = (likel2 - likel1) / H

        ls_gr_x.append(w)
        ls_gr_ac.append(dwk)

        sum = 0
        for v in w1.values():
            sum += 2 * v

        ls_gr_cp.append(dw[key])
        ls_gr_ll.append(likel1)

    plot.plot(ls_gr_x, ls_gr_ac)
    plot.plot(ls_gr_x, ls_gr_cp)
    # plot.plot(ls_gr_x, ls_gr_ll)
    plot.show()


if __name__ == '__main__':
    main()

