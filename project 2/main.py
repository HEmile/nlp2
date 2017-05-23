import sys
from helper import *
from earley import *
from libitg import *
from collections import defaultdict
from feature_helper import gradient, skip_bigrams
import pickle
import os
import matplotlib.pyplot as plot
import numpy as np
import random


LIMIT_TRANS_LENGTH = 3

PARTITION = 1

DATA_SET_INDEX = 0

SENTENCE_LENGTH = 10

BATCH_SIZE = 30

SGD_ITERATIONS = 10

LAMBDA_LR = 4

LAMBDA_R = 0.0001

GAMMA0 = 0.0001

USE_SPARSE_F = False

USE_LOAD_W = False

LOAD_W_PATH = 'wsparse1-50.pkl'

def main(parse=False, featurise=True, predict=False):
    chinese, english = read_data('data/training.zh-en')
    skip_dict = skip_bigrams(chinese)
    mn, mx = DATA_SET_INDEX * (len(chinese) // PARTITION), (DATA_SET_INDEX + 1) * (len(chinese) // PARTITION)
    chinese, english = chinese[mn: mx], english[mn: mx]
    lexicon, weights, ch_vocab, en_vocab, null_alligned = read_lexicon_ibm('lexicon')

    if USE_LOAD_W:
        with open(LOAD_W_PATH, 'rb') as f:
            w = pickle.load(f)
    else:
        w = defaultdict(float)

    if not os.path.exists('parses'):
        os.makedirs('parses')

    if not os.path.exists('features'):
        os.makedirs('features')

    print('Parsing sentences', mn, 'to', mx)
    likelihood = []
    count = 0
    g_batch = defaultdict(float)
    count_batch = 0
    best_likelihood = -sys.maxsize
    ls_gr_x = []
    ls_gr_y = []
    ls_gr_ac = []
    ls_gr_cp = []
    ls_exp_xy = []
    ls_exp_x = []
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

            if predict:
                print(en_src)

            dw, likel = gradient(dx, dxy, src_fsa, w, weights, skip_dict, index, featurise, LAMBDA_R, USE_SPARSE_F, predict)

            # if count % 10 == 0:
            #     H = 0.0001
            #     key, value = 'type:target_length', dw['type:target_length']
            #     wn = dict(w)
            #     wn[key] += H
            #     _, likel1, _, _ = gradient(dx, dxy, src_fsa, wn, weights, skip_dict, index, featurise, LAMBDA)
            #     dwk = (likel1 - likel) / H
            #
            #     ls_gr_x.append(w[key])
            #     ls_gr_y.append(dwk - value)
            #     ls_gr_ac.append(dwk)
            #     ls_gr_cp.append(value)

            # if count % 1000 == 0:
            #     plot.plot(ls_gr_x, ls_gr_ac)
            #     plot.plot(ls_gr_x, ls_gr_cp)
            #     plot.plot(ls_gr_x, ls_exp_x)
            #     plot.plot(ls_gr_x, ls_exp_xy)
            #     plot.show()

            likelihood.append(likel)
            if dw:
                for k, dwk in dw.items():
                    g_batch[k] += dwk
            if count_batch % BATCH_SIZE == 0:
                gammat = GAMMA0 * (1 / (1 + GAMMA0 * LAMBDA_LR * t))
                t += 1
                for k, dwk in g_batch.items():
                    w[k] += gammat * dwk / BATCH_SIZE
                g_batch = defaultdict(float)
            if count % 50 == 0:
                print(index)
                print(gammat)
                l = sum(likelihood) / count
                print(l)
                if l > best_likelihood:
                    modifier = 'sparse' if USE_SPARSE_F else ''
                    modifier += str(BATCH_SIZE) + '-' + str(LAMBDA_LR)
                    print(modifier)
                    with open('w'+modifier+str(iter)+'.pkl', 'wb') as f:
                        pickle.dump(dict(w), f)
                        best_likelihood = l
                likelihood = []
                count = 0

if __name__ == '__main__':
    main()

